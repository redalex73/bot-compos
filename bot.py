import discord
from discord.ext import commands
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# -------------------------------------------------
#               CARGA VARIABLES .env
# -------------------------------------------------
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")  # opcional

# -------------------------------------------------
#               ARCHIVO DE DATOS
# -------------------------------------------------
DATA_FILE = "Hero pool.xlsx"

# -------------------------------------------------
#           MODELO DE EMBEDDINGS LOCAL
# -------------------------------------------------
print("Cargando modelo de IA local...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(text):
    """Convierte texto en embedding vectorial"""
    vec = model.encode([text])[0]
    return np.array(vec, dtype="float32")

# -------------------------------------------------
#      LECTURA DEL EXCEL + CREACIÓN DEL ÍNDICE
# -------------------------------------------------
def load_data():
    print("Cargando Excel...")
    df = pd.read_excel(DATA_FILE)

    # Limpiamos columnas importantes
    df["Mapa"] = df["Mapa"].astype(str)
    df["Categoria"] = df["Categoria"].astype(str)

    # Crear embeddings de todos los mapas
    print("Generando embeddings...")
    matrix = np.array([embed(m) for m in df["Mapa"]])

    # Crear índice FAISS
    dim = matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    print("Datos cargados correctamente.")
    return df, index

df, faiss_index = load_data()

# -------------------------------------------------
#                    DISCORD BOT
# -------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Canal obligatorio (si se indica en .env)
ALLOWED_CHANNEL = int(CHANNEL_ID) if CHANNEL_ID else None

# -------------------------------------------------
#               FORMATEAR RESPUESTA
# -------------------------------------------------
def pretty_reply(map_row):
    """Devuelve un mensaje bonito con bullets"""
    return (
        f"**Mapa encontrado:**\n"
        f"• 🗺️ **Nombre:** {map_row['Mapa']}\n"
        f"• 🎮 **Categoría:** {map_row['Categoria']}\n"
        f"• 🛡️ **Tanque:** {map_row['TANQUE']}\n"
        f"• 🔥 **FDPS:** {map_row['FDPS']}\n"
        f"• 🎯 **HDPS:** {map_row['HDPS']}\n"
        f"• 💉 **Suport:** {map_row['SUPPORT']}"
    )

# -------------------------------------------------
#                 COMANDO /reload
# -------------------------------------------------
@bot.command()
async def reload(ctx):
    global df, faiss_index
    await ctx.send("🔄 Recargando datos...")

    df, faiss_index = load_data()

    await ctx.send("✅ Excel recargado correctamente.")

# -------------------------------------------------
#          MAPA ALEATORIO POR CATEGORÍA
# -------------------------------------------------
@bot.command()
async def maprandom(ctx, *, categoria: str):
    """Ej: !maprandom escolta"""
    if ALLOWED_CHANNEL and ctx.channel.id != ALLOWED_CHANNEL:
        return

    categoria = categoria.lower().strip()

    subset = df[df["Categoria"].str.lower() == categoria]

    if subset.empty:
        await ctx.send("❌ No tengo mapas en esa categoría.")
        return

    choice = subset.sample(1).iloc[0]
    await ctx.send(pretty_reply(choice))


# -------------------------------------------------
#   RESPUESTA AUTOMÁTICA SI LO MENCIONAN (@bot)
# -------------------------------------------------
@bot.event
async def on_message(message):
    # Ignorar mensajes del bot
    if message.author.bot:
        return

    # Respetar canal restringido si existe
    if ALLOWED_CHANNEL and message.channel.id != ALLOWED_CHANNEL:
        return

    # Mirar si mencionan al bot
    if bot.user in message.mentions:

        text = message.content.replace(f"<@{bot.user.id}>", "").strip()

        # Si pide mapa aleatorio
        if "aleatorio" in text.lower():
            cats = df["Categoria"].unique()
            cat = np.random.choice(cats)
            subset = df[df["Categoria"] == cat]
            choice = subset.sample(1).iloc[0]
            await message.channel.send(
                f"🎲 Mapa aleatorio de **{cat}**:\n" + pretty_reply(choice)
            )
            return

        # Búsqueda semántica del mapa
        user_vec = embed(text)
        D, I = faiss_index.search(np.array([user_vec]), k=1)

        map_row = df.iloc[I[0][0]]

        await message.channel.send(pretty_reply(map_row))

    # No olvidar procesar comandos (!reload, !maprandom)
    await bot.process_commands(message)

# -------------------------------------------------
#                     RUN
# -------------------------------------------------
print("Bot iniciado. Conectando a Discord...")
bot.run(DISCORD_TOKEN)


