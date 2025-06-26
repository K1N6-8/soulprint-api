
import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

ALLOWED_ARCHETYPES = {
    "Griot", "Kinara", "Ubuntu", "Jali", "Sankofa",
    "Imani", "Maji", "Nzinga", "Bisa", "Zamani",
    "Tamu", "Shujaa", "Ayo", "Ujamaa", "Kuumba"
}

model = AutoModelForSeq2SeqLM.from_pretrained("King-8/soulprint-generator")
tokenizer = AutoTokenizer.from_pretrained("King-8/soulprint-generator")
model.eval()

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(torch_device)

messages = {}
message_id_counter = 1

def generate_soulprint_message(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.95,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Soulprint API is running!"})

@app.route("/generate", methods=["POST"])
def generate():
    global message_id_counter
    data = request.get_json()

    age = data.get("age")
    archetype = data.get("archetype")
    location = data.get("location")
    profession = data.get("profession")

    if not all([age, archetype, location, profession]):
        return jsonify({"error": "age, archetype, location, and profession are required."}), 400

    if archetype not in ALLOWED_ARCHETYPES:
        return jsonify({
            "error": f"Invalid archetype. Choose one of: {', '.join(sorted(ALLOWED_ARCHETYPES))}"
        }), 400

    formatted_prompt = f"A {age}-year-old {archetype} from {location} who works as a {profession} says:"
    completion = generate_soulprint_message(formatted_prompt)

    message_id = str(message_id_counter)
    message_id_counter += 1

    messages[message_id] = {
        "id": message_id,
        "prompt": formatted_prompt,
        "completion": completion,
        "age": age,
        "archetype": archetype,
        "location": location,
        "profession": profession
    }

    return jsonify(messages[message_id]), 201

@app.route("/messages", methods=["GET"])
def get_all_messages_grouped():
    grouped = {}
    for msg in messages.values():
        arch = msg["archetype"]
        if arch not in grouped:
            grouped[arch] = []

        ordered_msg = {
            "id": msg["id"],
            "prompt": msg["prompt"],
            "completion": msg["completion"],
            "age": msg["age"],
            "archetype": msg["archetype"],
            "location": msg["location"],
            "profession": msg["profession"]
        }

        grouped[arch].append(ordered_msg)

    return jsonify(grouped), 200

@app.route("/messages/<message_id>", methods=["GET"])
def get_message(message_id):
    if message_id in messages:
        return jsonify(messages[message_id]), 200
    else:
        return jsonify({"error": "Message not found."}), 404

@app.route("/messages/<message_id>", methods=["PATCH"])
def update_message(message_id):
    if message_id not in messages:
        return jsonify({"error": "Message not found."}), 404

    data = request.get_json()
    new_age = data.get("age")
    new_archetype = data.get("archetype")
    new_location = data.get("location")
    new_profession = data.get("profession")

    if new_archetype and new_archetype not in ALLOWED_ARCHETYPES:
        return jsonify({
            "error": f"Invalid archetype. Choose one of: {', '.join(sorted(ALLOWED_ARCHETYPES))}"
        }), 400

    age = new_age or messages[message_id]["age"]
    archetype = new_archetype or messages[message_id]["archetype"]
    location = new_location or messages[message_id]["location"]
    profession = new_profession or messages[message_id]["profession"]

    new_prompt = f"A {age}-year-old {archetype} from {location} who works as a {profession} says:"
    new_completion = generate_soulprint_message(new_prompt)

    messages[message_id].update({
        "prompt": new_prompt,
        "completion": new_completion,
        "age": age,
        "archetype": archetype,
        "location": location,
        "profession": profession
    })

    return jsonify(messages[message_id]), 200

@app.route("/messages/<message_id>", methods=["DELETE"])
def delete_message(message_id):
    if message_id in messages:
        del messages[message_id]
        return jsonify({"message": "Deleted successfully."}), 200
    else:
        return jsonify({"error": "Message not found."}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
