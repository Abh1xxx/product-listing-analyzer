"use strict";
const byId = id => document.getElementById(id);
const examples = {
  shirt: ["Northstar Blue Cotton T-Shirt for Everyday Wear", "This lightweight cotton T-shirt has a comfortable fit for everyday wear. The soft fabric and classic crew neck make it easy to pair with casual outfits."],
  headphones: ["SoundPeak Black Wireless Headphones with Noise Isolation", "Enjoy clear audio with these lightweight wireless headphones. The padded ear cups provide a comfortable fit, while the rechargeable battery supports long listening sessions."]
};
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}
function definition(parent, label, value) {
  parent.append(node("dt", label), node("dd", value));
}
function table(headers, rows) {
  const wrap = node("div", undefined, "table-wrap"), table = node("table"), head = node("thead"), tr = node("tr");
  headers.forEach(header => tr.append(node("th", header))); head.append(tr); table.append(head);
  const body = node("tbody");
  rows.forEach(row => { const line = node("tr"); row.forEach(value => line.append(node("td", value ?? ""))); body.append(line); });
  table.append(body); wrap.append(table); return wrap;
}
byId("example").addEventListener("change", event => {
  const [title, description] = examples[event.target.value] || ["", ""];
  byId("title").value = title; byId("description").value = description;
});
let datasetLoaded = false;
document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(tab => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-pressed", String(tab === button)); });
  document.querySelectorAll(".page").forEach(page => { page.hidden = page.id !== button.dataset.page; });
  if (button.dataset.page === "dataset" && !datasetLoaded) loadDataset();
}));
async function loadDataset() {
  const status = byId("dataset-status"); status.textContent = "Loading dataset…";
  try {
    const response = await fetch("/api/dataset");
    if (!response.ok) throw new Error("Dataset could not be loaded. Select this tab again to retry.");
    const data = await response.json(), target = byId("dataset-details"); target.replaceChildren();
    status.textContent = `${data.records} product records · ${data.columns.length} columns · ${data.duplicates_removed} exact duplicate records removed`;
    target.append(node("h3", "Available columns & types"), table(["Column", "Data type"], Object.entries(data.types)), node("h3", "Detected fields"));
    const fields = node("dl"); Object.entries(data.fields).forEach(([name, value]) => definition(fields, name, Array.isArray(value) ? value.join(", ") : value || "Not available")); target.append(fields);
    target.append(node("h3", "First five records"), table(data.columns, data.preview.map(row => data.columns.map(column => row[column])))); datasetLoaded = true;
  } catch (error) { status.textContent = error.message; }
}
byId("listing-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = byId("submit"), status = byId("status");
  const title = byId("title").value.trim(), description = byId("description").value.trim();
  if (!title || !description) { status.textContent = "Enter both a title and a description."; status.className = "error"; return; }
  button.disabled = true; status.className = ""; status.textContent = "Analyzing your listing…"; byId("results").hidden = true;
  try {
    const response = await fetch("/api/analyze", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({title,description})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Analysis failed. Please try again.");
    render(data); status.textContent = "Analysis complete."; byId("results").hidden = false; byId("results").focus();
  } catch (error) { status.textContent = error instanceof SyntaxError ? "The service returned an unexpected response. Please try again." : error.message; status.className = "error"; }
  finally { button.disabled = false; }
});
function render(data) {
  byId("total").textContent = data.quality_score.total;
  const components = byId("components"); components.replaceChildren();
  Object.entries(data.quality_score.components).forEach(([name, points]) => { const card = node("div", name, "component"), value = node("strong", points); value.append(node("small", " / 20")); card.append(value); components.append(card); });
  byId("keywords").replaceChildren(data.keywords.length ? table(["Keyword", "TF-IDF score"], data.keywords.map(([word, score]) => [word, score.toFixed(4)])) : node("p", "No in-vocabulary terms were found."));
  byId("similarity").textContent = data.cosine_similarity === null ? "" : `Title–description cosine similarity: ${data.cosine_similarity.toFixed(3)} (lexical overlap)`;
  const attributes = byId("attributes"); attributes.replaceChildren(); Object.entries(data.attributes).forEach(([name, values]) => definition(attributes, name, values.join(", ") || "No match detected"));
  const advice = byId("advice"); advice.replaceChildren();
  (data.problems.length ? data.problems : ["No basic title-length, description-length, or repetition issue was detected."]).forEach(value => advice.append(node("p", value, "notice")));
  data.suggestions.forEach(value => advice.append(node("p", value, "suggestion")));
  const cleaned = byId("cleaned"); cleaned.replaceChildren(); ["original_title", "original_description", "cleaned_title", "cleaned_description"].forEach(key => definition(cleaned, key.replaceAll("_", " "), data[key]));
  const counts = byId("counts"); counts.replaceChildren(); ["word_count", "unique_word_count", "character_count", "sentence_count"].forEach(key => counts.append(node("span", `${key.replaceAll("_", " ")}: ${data[key]}`)));
  const tokens = byId("tokens"); tokens.replaceChildren(); ["tokens_before_stopword_removal", "tokens_after_stopword_removal", "stemmed_tokens", "lemmatized_tokens"].forEach(key => definition(tokens, key.replaceAll("_", " "), data[key].join(", ")));
  const phrases = byId("phrases"); phrases.replaceChildren();
  definition(phrases, "Repeated words", Object.entries(data.repeated_words).map(([word, count]) => `${word} (${count})`).join(", ") || "None");
  ["word_frequencies", "bigrams", "trigrams"].forEach(key => { phrases.append(node("h3", key.replaceAll("_", " ")), table(["Term / phrase", "Count"], data[key])); });
  const rules = byId("rules"); rules.replaceChildren(); Object.entries(data.quality_score.rules).forEach(([name, value]) => definition(rules, name, value));
}
