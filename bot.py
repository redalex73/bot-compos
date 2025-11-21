# bot.py
import os
import random
import pickle
import time
import re

import pandas as pd
import numpy as np
import faiss

import discord
from discord.ext import commands

from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Config ----------------
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Si quieres restringir el bot a un canal concreto, pon el ID en .env (ej: 123456789012345678)
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()

# Ruta al Excel que subiste (ya está en tu entorno):
DATA_FILE = "/mnt/data/Hero pool.xlsx"

# Caché de embeddings / faiss
EMB_PKL = "map_embeds.pkl"
FAISS_IDX = "map_faiss.index"

# Cliente OpenAI
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# Palabras clave para pedir mapa aleatorio
RANDOM_KEYWORDS = [
    "aleatorio", "random", "dame un mapa", "elige un mapa", "mapa aleatorio",
    "dame un mapita", "mapita", "elige mapa", "sorprendeme",
    "mover el robot", "robot", "payload", "escolta", "escort", "mover"
]

# Sinónimos / heurística para tipos de mapa (puedes ampliarlo)
TYPE_SYNONYMS = {
    "mover el robot": "Push",
    "robot": "Push",
    "payload": "Escort",
    "escoltar": "Escort",
    "control": "Control",
    "híbrido": "Hybrid",
    "hybrid": "Hybrid",
    "push": "Push",
    "captura": "Control",
    "escort": "Escort"
}

# -------------- Helpers para Excel --------------
def load_raw_df(path=DATA_FILE):
    # Leemos sin suponer cabeceras (tu Excel tiene una fila inicial con títulos)
    df = pd.read_excel(path, header=None, dtype=str)
    df = df.fillna("")
    if df.shape[0] < 1:
        raise ValueError("El Excel parece vacío.")
    header = df.iloc[0].tolist()
    body = df.iloc[1:].copy()
    body.columns = header
    body = body.reset_index(drop=True)
    # eliminar columnas vacías completamente
    body = body.loc[:, (body != "").any(axis=0)]
    return body

def detect_map_column(df):
    candidates = [c for c in df.columns if str(c).strip().lower() in ("mapas", "mapa", "map", "maps", "mapas ")]
    if candidates:
        return candidates[0]
    # fallback: primera columna con datos
    for c in df.columns:
        if df[c].astype(str).str.strip().replace("", np.nan).notna().any():
            return c
    return df.columns[0]

def build_map_list(df, map_col):
    maps = df[map_col].astype(str).tolist()
    maps = [m.strip() for m in maps if m.strip() != ""]
    # sacar duplicados manteniendo orden
    seen = set()
    unique = []
    for m in maps:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique

# -------------- Embeddings / FAISS --------------
def generate_and_save_embeddings(maps):
    print("Generando embeddings para mapas (OpenAI) ...")
    vectors = []
    for m in maps:
        resp = client_ai.embeddings.create(model="text-embedding-3-small", input=m)
        emb = resp.data[0].embedding
        vectors.append(np.array(emb, dtype="float32"))
        time.sleep(0.01)
    vecs = np.array(vectors).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, FAISS_IDX)
    with open(EMB_PKL, "wb") as f:
        pickle.dump({"maps": maps, "vecs_shape": vecs.shape}, f)
    print("Embeddings guardados.")

def load_embeddings_or_build(maps):
    if os.path.exists(EMB_PKL) and os.path.exists(FAISS_IDX):
        try:
            with open(EMB_PKL, "rb") as f:
                meta = pickle.load(f)
            index = faiss.read_index(FAISS_IDX)
            if len(meta.get("maps", [])) == len(maps):
                return meta["maps"], index
            else:
                print("La lista de mapas cambió — regenerando embeddings...")
        except Exception as e:
            print("Error leyendo cache de embeddings, regenerando:", e)
    generate_and_save_embeddings(maps)
    with open(EMB_PKL, "rb") as f:
        meta = pickle.load(f)
    index = faiss.read_index(FAISS_IDX)
    return meta["maps"], index

# -------------- Búsqueda semántica --------------
def find_closest_map(query, maps, index, topk=1):
    resp = client_ai.embeddings.create(model="text-embedding-3-small", input=query)
    qemb = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    D, I = index.search(qemb, topk)
    i = int(I[0][0])
    return maps[i], float(D[0][0])

# -------------- Formateo respuesta (solo roles) --------------
def strip_player_names(val):
    # Quitar paréntesis con nombres al final: "Zarya (Max)" -> "Zarya"
    val = re.sub(r"\s*\([^\)]{1,30}\)\s*$", "", val)
    # remover nombres solapados por "/" si parecen jugadores? No tocar, dejamos opciones "Cassidy/Sojourn"
    val = " ".join(val.split())
    return val

def format_map_response(df, map_name, map_col):
    matches = df[df[map_col].astype(str).str.strip().str.lower() == map_name.strip().lower()]
    if matches.empty:
        matches = df[df[map_col].astype(str).str.lower().str.contains(map_name.strip().lower())]
    if matches.empty:
        return f"No encontré información para el mapa **{map_name}**."

    row = matches.iloc[0]

    role_columns_candidates = ["Tank", "tank", "FDPS", "fdps", "HDPS", "hdps",
                               "Flex Supp", "Flex Support", "Flex", "Main Supp",
                               "Main Support", "Support", "Supp", "Extra", "Mains", "mains"]
    found_roles = []
    for col in df.columns:
        col_norm = str(col).strip()
        if any(col_norm.lower() == c.lower() for c in role_columns_candidates):
            found_roles.append(col_norm)
    if not found_roles:
        all_cols = list(df.columns)
        idx = all_cols.index(map_col)
        found_roles = all_cols[idx+1: idx+7]

    lines = [f"**Mapa: {map_name}**"]
    for role_col in found_roles:
        val = str(row.get(role_col, "")).strip()
        if val:
            val_clean = strip_player_names(val)
            lines.append(f"• **{role_col}**: {val_clean}")
    return "\n".join(lines)

# -------------- Intent detection --------------
def is_random_request(text):
    t = text.lower()
    for kw in RANDOM_KEYWORDS:
        if kw in t:
            return True
    return False

def guess_type_from_text(text):
    t = text.lower()
    for key, tipo in TYPE_SYNONYMS.items():
        if key in t:
            return tipo
    for tipo in ["Escort", "Control", "Assault", "Hybrid", "Push"]:
        if tipo.lower() in t:
            return tipo
    return None

# -------------- Preparación inicial --------------
def prepare():
    df = load_raw_df(DATA_FILE)
    map_col = detect_map_column(df)
    maps = build_map_list(df, map_col)
    maps_embeds, index = load_embeddings_or_build(maps)
    return df, map_col, maps, index

print("Preparando bot y datos...")
DF, MAP_COL, MAPS, INDEX = prepare()
print(f"Listo. Mapas detectados: {len(MAPS)} (ej: {MAPS[:6]})")

# -------------- Bot Discord --------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user} (id {bot.user.id})")

@bot.command(name="reload")
async def _reload(ctx):
    if ctx.author.guild_permissions.manage_guild or await bot.is_owner(ctx.author):
        await ctx.send("🔄 Recargando datos... (esto regenerará embeddings si la lista de mapas cambió)")
        try:
            global DF, MAP_COL, MAPS, INDEX
            DF, MAP_COL, MAPS, INDEX = prepare()
            await ctx.send("✅ Datos recargados correctamente.")
        except Exception as e:
            await ctx.send(f"❌ Error recargando: {e}")
    else:
        await ctx.send("No tienes permisos para ejecutar este comando.")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Si CHANNEL_ID configurado, solo responder en ese canal
    if CHANNEL_ID:
        try:
            if str(message.channel.id) != str(CHANNEL_ID):
                return
        except Exception:
            pass

    if bot.user in message.mentions:
        content = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if content == "":
            await message.channel.send("¿Sí? ¿Qué necesitas?")
            return

        # petición de mapa aleatorio
        if is_random_request(content):
            tipo = guess_type_from_text(content)
            if tipo:
                chosen = random.choice(MAPS)
                resp = format_map_response(DF, chosen, MAP_COL)
                await message.channel.send(f"**Mapa aleatorio (tipo: {tipo})**:\n{resp}")
                return
            else:
                chosen = random.choice(MAPS)
                resp = format_map_response(DF, chosen, MAP_COL)
                await message.channel.send(f"**Mapa aleatorio**:\n{resp}")
                return

        # búsqueda semántica
        try:
            best_map, distance = find_closest_map(content, MAPS, INDEX)
            formatted = format_map_response(DF, best_map, MAP_COL)
            await message.channel.send(formatted)
            return
        except Exception as e:
            await message.channel.send(f"No he podido procesar tu petición (error interno: {e})")
            return

    await bot.process_commands(message)

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise ValueError("Pon tu token de Discord en .env (DISCORD_TOKEN).")
    if not OPENAI_API_KEY:
        raise ValueError("Pon tu API key de OpenAI en .env (OPENAI_API_KEY).")

    bot.run(DISCORD_TOKEN)
