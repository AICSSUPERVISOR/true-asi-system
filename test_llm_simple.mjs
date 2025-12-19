const response = await fetch(process.env.BUILT_IN_FORGE_API_URL + "/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${process.env.BUILT_IN_FORGE_API_KEY}`
  },
  body: JSON.stringify({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say 'LLM WORKS' if you can read this" }]
  })
});
const data = await response.json();
console.log("Status:", response.status);
console.log("Response:", data.choices?.[0]?.message?.content || data);
