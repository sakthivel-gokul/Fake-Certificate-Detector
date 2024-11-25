const fs = require('fs');

(async () => {
    const fetch = (await import('node-fetch')).default;

    // Hugging Face API setup
    const HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large";
    const HUGGINGFACE_API_KEY = "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";

    async function queryHuggingFace(filename) {
        const data = fs.readFileSync(filename);
        const encodedImage = Buffer.from(data).toString("base64");
        const payload = { inputs: encodedImage };

        const response = await fetch(HUGGINGFACE_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${HUGGINGFACE_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Hugging Face API error: ${response.statusText}`);
        }

        const result = await response.json();
        return result[0]?.generated_text || "";
    }

    // Groq API setup
    const GROQ_API_URL = "https://api.groq.com/v1/classify-text"; // Replace with actual endpoint
    const GROQ_API_KEY = "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";

    async function categorizeWithGroq(hfOutputText) {
        const prompt = `
            Given the description '${hfOutputText}', classify the issue into one of the following categories:
            "Cleanliness", "Staff Behavior", "Punctuality", "Water Availability", "Food Quality", "Security", 
            "Seating and Comfort", "Washroom Facilities", "Noise Disturbance", "Technical Malfunctions".

            Do not give anything other than the exact category output.
        `;

        const response = await fetch(GROQ_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${GROQ_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: "llama3-8b-8192",
                messages: [{ role: "user", content: prompt }]
            })
        });

        if (!response.ok) {
            throw new Error(`Groq API error: ${response.statusText}`);
        }

        const result = await response.json();
        return result.choices[0]?.message?.content || "";
    }

    // Workflow
    (async () => {
        try {
            const hfOutputText = await queryHuggingFace("E:/Document_detection/hqdefault.jpg");
            console.log("Hugging Face Output:", hfOutputText);

            const groqCategory = await categorizeWithGroq(hfOutputText);
            console.log("Groq Categorization:", groqCategory);
        } catch (error) {
            console.error("Error:", error.message);
        }
    })();
})();