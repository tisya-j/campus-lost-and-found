import streamlit as st
import clip
import torch
import sqlite3
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- SETUP ----------------
st.set_page_config(page_title="Campus Lost & Found", layout="centered")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Smaller model uses less RAM
model, preprocess = clip.load("RN50", device=device)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("items.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        location TEXT,
        contact TEXT,
        image_path TEXT,
        embedding BLOB
    )
''')
conn.commit()

# ---------------- UTILS ----------------
def encode_image(image_path):
    # Resize to reduce memory usage
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image).cpu().numpy()

def encode_text(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        return model.encode_text(tokens).cpu().numpy()

def save_item(description, location, contact, image_path, embedding):
    embedding_blob = embedding.astype(np.float32).tobytes()
    c.execute(
        "INSERT INTO items (description, location, contact, image_path, embedding) VALUES (?, ?, ?, ?, ?)",
        (description, location, contact, image_path, embedding_blob)
    )
    conn.commit()

def delete_item(item_id):
    c.execute("DELETE FROM items WHERE id=?", (item_id,))
    conn.commit()

# ---------------- UI ----------------
st.title("Campus Lost & Found 🏫")

tab1, tab2, tab3 = st.tabs(["Report Lost Item", "Report Found Item", "Search Items"])

# ---------------- FOUND ITEM ----------------
with tab2:
    st.subheader("Report Found Item")
    found_text = st.text_input("Describe the found item", key="found_text")
    found_image = st.file_uploader("Upload image (optional)", type=["jpg", "png"], key="found_image")
    found_location = st.text_input("Location on campus (optional)", key="found_location")
    found_contact = st.text_input("Your contact info (optional)", key="found_contact")

    if st.button("Save Found Item", key="save_found"):
        if not found_text and not found_image:
            st.error("Provide at least text or image.")
            st.stop()

        image_path = None
        if found_image:
            image_path = os.path.join(UPLOAD_DIR, found_image.name)
            with open(image_path, "wb") as f:
                f.write(found_image.read())
            embedding = encode_image(image_path)
        else:
            embedding = encode_text(found_text)

        save_item(found_text, found_location, found_contact, image_path, embedding)
        st.success("Found item saved successfully!")

# ---------------- LOST ITEM ----------------
with tab1:
    st.subheader("Report Lost Item")
    lost_text = st.text_input("Describe the lost item (optional)", key="lost_text")
    lost_image = st.file_uploader("Upload image (optional)", type=["jpg", "png"], key="lost_image")
    lost_location = st.text_input("Location on campus (optional)", key="lost_location")

    if st.button("Find Matches", key="find_matches"):
        if not lost_text and not lost_image:
            st.error("Please provide text or an image.")
            st.stop()

        if lost_image:
            image_path = os.path.join(UPLOAD_DIR, lost_image.name)
            with open(image_path, "wb") as f:
                f.write(lost_image.read())
            query_embedding = encode_image(image_path)
        else:
            query_embedding = encode_text(lost_text)

        # Fetch items with embeddings one by one to save memory
        c.execute("SELECT id, description, location, contact, image_path, embedding FROM items")
        rows = c.fetchall()
        matches = []
        for row in rows:
            item_id, desc, loc, contact, image_path, embedding_blob = row
            if embedding_blob:
                emb = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
                score = cosine_similarity(query_embedding, emb)[0][0]
                matches.append((score, {
                    "id": item_id,
                    "description": desc,
                    "location": loc,
                    "contact": contact,
                    "image": image_path
                }))
        matches.sort(reverse=True, key=lambda x: x[0])

        st.subheader("🔍 Possible Matches")
        found_any = False
        for score, item in matches[:5]:
            if score > 0.75:
                found_any = True
                st.write(f"**Description:** {item['description']}")
                if item["location"]:
                    st.write(f"**Location:** {item['location']}")
                if item["contact"]:
                    st.write(f"**Contact:** {item['contact']}")
                if item["image"]:
                    st.image(item["image"], width=200)
                st.write(f"**Similarity:** {round(score * 100, 2)}%")
                st.divider()
        if not found_any:
            st.info("No strong matches found yet.")

# ---------------- SEARCH TAB ----------------
with tab3:
    st.subheader("Search Items by keyword or location")
    search_term = st.text_input("Enter search keyword or location", key="search_bar")
    if search_term:
        c.execute("SELECT id, description, location, contact, image_path FROM items")
        rows = c.fetchall()
        results_found = False
        for row in rows:
            item_id, desc, loc, contact, image_path = row
            if search_term.lower() in desc.lower() or (loc and search_term.lower() in loc.lower()):
                results_found = True
                st.write(f"**Description:** {desc}")
                if loc:
                    st.write(f"**Location:** {loc}")
                if contact:
                    st.write(f"**Contact:** {contact}")
                if image_path:
                    st.image(image_path, width=200)
                st.divider()
        if not results_found:
            st.info("No items match your search.")
