import glob
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, GenerateVideosConfig


# =====================================================
# CONFIG
# =====================================================

@dataclass(frozen=True)
class AppConfig:
    project_id: str
    location: str = "us-central1"
    model_video: str = "veo-3.1-generate-001"
    model_analise: str = "gemini-2.5-pro"
    arquivo_style_bible: str = "style_bible.txt"
    arquivo_analise: str = "proximo_episodio.txt"
    timeout_operacao_segundos: int = 900
    intervalo_polling_segundos: int = 10


def carregar_config() -> AppConfig:
    project_id = os.getenv("PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError(
            "Variável de ambiente PROJECT_ID não definida. "
            "Exemplo: export PROJECT_ID='seu-projeto'"
        )
    return AppConfig(project_id=project_id)


# =====================================================
# CLIENTE
# =====================================================


def criar_cliente(cfg: AppConfig) -> genai.Client:
    return genai.Client(vertexai=True, project=cfg.project_id, location=cfg.location)


# =====================================================
# UTILITÁRIOS DE VÍDEO
# =====================================================


def extrair_ultimo_frame(video_path: str, frame_output_path: str = "ultimo_frame.png") -> Optional[str]:
    print(f"🔍 Extraindo o último frame de: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir o vídeo.")
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("❌ Erro: O vídeo não tem frames.")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro: Não foi possível ler o último frame.")
            return None

        cv2.imwrite(frame_output_path, frame)
        print(f"✅ Último frame salvo como: {frame_output_path}")
        return frame_output_path
    finally:
        cap.release()


def parse_tempo_para_segundos(tempo_str: str) -> float:
    txt = tempo_str.strip().lower()
    if txt.endswith("ms"):
        return float(txt[:-2].strip()) / 1000.0
    if txt.endswith("s"):
        return float(txt[:-1].strip())

    valor = float(txt)
    if valor >= 1000:
        raise ValueError("Valor ambíguo sem unidade. Use sufixo 'ms' ou 's'.")
    return valor


def extrair_frame_em_tempo(video_path: str, tempo_segundos: float, frame_output_path: str = "frame_especifico.png") -> Optional[str]:
    print(f"🔍 Extraindo frame em {tempo_segundos:.3f}s de: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir o vídeo.")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracao_total = total_frames / fps if fps > 0 else 0

        if tempo_segundos < 0 or tempo_segundos > duracao_total:
            print(
                f"⚠️ Tempo {tempo_segundos:.3f}s inválido. "
                f"Duração do vídeo: {duracao_total:.3f}s. Usando último frame."
            )
            return extrair_ultimo_frame(video_path)

        frame_idx = int(tempo_segundos * fps) if fps > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro: Não foi possível ler o frame no tempo solicitado. Usando último frame.")
            return extrair_ultimo_frame(video_path)

        cv2.imwrite(frame_output_path, frame)
        print(f"✅ Frame em {tempo_segundos:.3f}s salvo como: {frame_output_path}")
        return frame_output_path
    finally:
        cap.release()


# =====================================================
# PROMPT / ANÁLISE
# =====================================================


def montar_system_instruction(style_bible: str) -> str:
    bible = style_bible if style_bible else "Found footage, handheld, VHS, sem cortes."
    return f"""
Você é um diretor especialista em found footage de terror.
Siga continuidade absoluta entre último frame e nova cena.

REGRAS:
- Não introduza personagens/objetos não implícitos.
- Respeite a bíblia visual integralmente.
- Entregue saída no formato pedido.

BÍBLIA DE ESTILO:
--- INÍCIO ---
{bible}
--- FIM ---
""".strip()


def montar_user_instruction() -> str:
    return """
Analise o vídeo e o frame enviados e responda APENAS com JSON válido (sem markdown), no schema:
{
  "continuity_checks": ["string"],
  "branches": {
    "A": "string",
    "B": "string",
    "C": "string"
  },
  "scores": {
    "A": {"consistency": 1-10, "effectiveness": 1-10},
    "B": {"consistency": 1-10, "effectiveness": 1-10},
    "C": {"consistency": 1-10, "effectiveness": 1-10}
  },
  "winner": {
    "branch": "A|B|C",
    "justification": "2-3 frases"
  },
  "timeline": {
    "0_3s": "string",
    "3_6s": "string",
    "6_8s": "string"
  },
  "veo_prompt_en": "string"
}
""".strip()


def analisar_video_e_gerar_prompt(client: genai.Client, cfg: AppConfig, video_path: str, frame_path: Optional[str] = None) -> None:
    print("\n⏳ Iniciando análise do vídeo com Gemini...")

    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
    except Exception as e:
        print(f"❌ Falha ao ler o vídeo: {e}")
        return

    if frame_path is None or not os.path.exists(frame_path):
        frame_path = extrair_ultimo_frame(video_path)

    frame_bytes = None
    if frame_path:
        try:
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
        except Exception as e:
            print(f"⚠️ Não foi possível ler o frame: {e}")

    style_bible = ""
    if os.path.exists(cfg.arquivo_style_bible):
        style_bible = Path(cfg.arquivo_style_bible).read_text(encoding="utf-8")

    parts = [
        types.Part.from_text(text=montar_user_instruction()),
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
    ]
    if frame_bytes:
        parts.append(types.Part.from_bytes(data=frame_bytes, mime_type="image/png"))

    try:
        response = client.models.generate_content(
            model=cfg.model_analise,
            contents=types.Content(parts=parts, role="user"),
            config=GenerateContentConfig(system_instruction=montar_system_instruction(style_bible)),
        )
        texto = response.text or ""
        # valida JSON antes de persistir; se vier inválido, persiste cru para debug
        try:
            parsed = json.loads(texto)
            texto = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            print("⚠️ Resposta não veio em JSON válido; salvando conteúdo bruto para inspeção.")
        print(texto)
        Path(cfg.arquivo_analise).write_text(texto, encoding="utf-8")
        print(f"💾 Análise salva em '{cfg.arquivo_analise}'")
    except Exception as e:
        print(f"❌ Erro ao consultar o Gemini: {e}")


def extrair_prompt_english(conteudo: str) -> Optional[str]:
    """Extrai prompt em inglês de JSON (preferencial) ou bloco markdown (fallback)."""

    def _from_dict(data: dict) -> Optional[str]:
        candidatos = [
            data.get("veo_prompt_en"),
            data.get("prompt_en"),
            data.get("english_prompt"),
        ]
        for value in candidatos:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    # 1) JSON direto no arquivo
    try:
        data = json.loads(conteudo)
        if isinstance(data, dict):
            prompt = _from_dict(data)
            if prompt:
                return prompt
    except json.JSONDecodeError:
        pass

    # 2) JSON cercado por texto; tenta capturar o maior objeto possível
    match_json = re.search(r"\{[\s\S]*\}", conteudo)
    if match_json:
        try:
            data = json.loads(match_json.group(0))
            if isinstance(data, dict):
                prompt = _from_dict(data)
                if prompt:
                    return prompt
        except json.JSONDecodeError:
            pass

    # 3) Fallback legado para bloco markdown
    match_code = re.search(r"```(?:english|en)?\s*\n([\s\S]*?)\n```", conteudo, flags=re.IGNORECASE)
    if match_code:
        prompt = match_code.group(1).strip()
        if prompt:
            return prompt

    return None


def aguardar_operacao(client: genai.Client, operation, cfg: AppConfig):
    inicio = time.time()
    while not operation.done:
        if time.time() - inicio > cfg.timeout_operacao_segundos:
            raise TimeoutError(f"Timeout após {cfg.timeout_operacao_segundos}s")
        print("🔄 Aguardando...")
        time.sleep(cfg.intervalo_polling_segundos)
        operation = client.operations.get(operation)
    return operation


def gerar_video(client: genai.Client, cfg: AppConfig, prompt: str, arquivo_saida: str):
    seed = int(datetime.now().timestamp())
    operation = client.models.generate_videos(
        model=cfg.model_video,
        prompt=prompt,
        config=GenerateVideosConfig(number_of_videos=1, aspect_ratio="16:9", enhance_prompt=True, seed=seed),
    )
    operation = aguardar_operacao(client, operation, cfg)

    if not operation.response:
        print("❌ Erro na geração:", operation.error)
        return

    generated_video = operation.result.generated_videos[0]
    with open(arquivo_saida, "wb") as f:
        f.write(generated_video.video.video_bytes)
    print(f"✅ Vídeo salvo como '{arquivo_saida}'")


opcoes = {
    "cenario": {
        "1": "Abandoned hospital corridor with flickering lights and bloody handprints",
        "2": "Dense forest at night, only flashlight beam, old abandoned cabin in the distance",
        "3": "Dark basement with children's toys scattered, occult symbols on the floor",
        "4": "Deserted motel hallway, doors creaking, neon sign buzzing outside",
        "5": "Underground parking garage, flickering fluorescent lights, water dripping",
        "6": "Old Victorian mansion, dust floating, cobwebs, grandfather clock ticking loudly",
    },
    "camera": {
        "1": "shaky handheld POV camera, very unstable, sudden whips",
        "2": "static tripod shot, but camera is bumped and falls",
        "3": "CCTV security camera angle, grainy, timestamp overlay",
        "4": "body cam attached to someone running, heavy breathing",
        "5": "drone shot slowly descending into chaos",
        "6": "cell phone vertical video, dropped and cracking",
    },
    "efeitos": {
        "1": "green night vision mode, glitching with VHS tracking artifacts",
        "2": "VHS grain, scanlines, slight chromatic aberration",
        "3": "black and white, high contrast, overexposed",
        "4": "lens flare, dust on lens, occasional focus pulls",
        "5": "infrared thermal vision (red/blue heat map)",
        "6": "analog horror style, sudden cuts to black, distorted subtitles",
    },
    "audio": {
        "1": "heavy breathing and panicked whispers",
        "2": "distant echoing footsteps, metal clanging",
        "3": "radio static with distorted voices",
        "4": "inhuman scream from darkness, then silence",
        "5": "heartbeat thumping, growing louder",
        "6": "children's laughter echoing, getting closer",
    },
    "elementos": {
        "1": "shadow figure at the end of the hall, tilting head",
        "2": "creature crawling on the ceiling, backwards",
        "3": "door slamming shut by itself, then slowly opening",
        "4": "pair of pale feet walking towards the fallen camera",
        "5": "blood dripping from ceiling, spelling words",
        "6": "old TV turning on by itself, showing static",
    },
}


def mostrar_menu(titulo, itens):
    print(f"\n--- {titulo} ---")
    for chave, desc in itens.items():
        print(f"  {chave}. {desc}")
    print("  (deixe em branco para pular)")


def selecionar_opcoes(titulo, itens):
    mostrar_menu(titulo, itens)
    escolhas = input("Digite os números desejados (separados por vírgula): ").strip()
    if not escolhas:
        return []
    indices = [x.strip() for x in escolhas.split(",") if x.strip() in itens]
    return [itens[i] for i in indices]


def salvar_style_bible(cfg: AppConfig, partes_prompt):
    with open(cfg.arquivo_style_bible, "w", encoding="utf-8") as f:
        f.write("GÊNERO: Found Footage Horror.\n")
        f.write("CÂMERA: " + (partes_prompt[1] if len(partes_prompt) > 1 else "Handheld, shaky") + "\n")
        f.write("EFEITOS: " + (partes_prompt[2] if len(partes_prompt) > 2 else "VHS grain, night vision") + "\n")
        f.write("AMBIENTE: " + (partes_prompt[0] if len(partes_prompt) > 0 else "Abandoned hospital") + "\n")
        f.write("ÁUDIO: Ambiente com estática e respiração ofegante.\n")


def main():
    try:
        cfg = carregar_config()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    client = criar_cliente(cfg)

    while True:
        print("\n" + "=" * 60)
        print("🎬 GERADOR DE FOUND FOOTAGE HORROR - VEO 3.1 + GEMINI PRO")
        print("=" * 60)
        print("1. Gerar um novo vídeo")
        print("2. Analisar um vídeo e sugerir o próximo episódio")
        print("3. Gerar próximo episódio a partir da análise salva")
        print("4. Sair")
        escolha = input("Escolha uma opção: ").strip()

        if escolha == "1":
            cenarios = selecionar_opcoes("CENÁRIO", opcoes["cenario"])
            while not cenarios:
                print("⚠️ É necessário escolher pelo menos um cenário.")
                cenarios = selecionar_opcoes("CENÁRIO", opcoes["cenario"])

            camera = selecionar_opcoes("ESTILO DE CÂMERA", opcoes["camera"])
            efeitos = selecionar_opcoes("EFEITOS VISUAIS", opcoes["efeitos"])
            audio = selecionar_opcoes("ÁUDIO / AMBIENTE", opcoes["audio"])
            elementos = selecionar_opcoes("ELEMENTOS DE TERROR", opcoes["elementos"])

            partes_prompt = [f"Setting: {'; '.join(cenarios)}."]
            if camera:
                partes_prompt.append(f"Camera style: {', '.join(camera)}.")
            if efeitos:
                partes_prompt.append(f"Visual effects: {', '.join(efeitos)}.")
            if audio:
                partes_prompt.append(f"Audio: {', '.join(audio)}.")
            if elementos:
                partes_prompt.append(f"Horror elements: {', '.join(elementos)}.")

            prompt_base = (
                "Found footage horror video, POV handheld camera, extreme realism. "
                + " ".join(partes_prompt)
                + " Grainy, low-light, analog horror aesthetic, video ends in static."
            )

            salvar_style_bible(cfg, partes_prompt)
            print(prompt_base)
            input("Pressione ENTER para gerar o vídeo...")
            gerar_video(client, cfg, prompt_base, "found_footage_horror.mp4")

        elif escolha == "2":
            videos = [f for f in os.listdir() if f.endswith(".mp4")]
            if not videos:
                print("❌ Nenhum vídeo encontrado.")
                continue
            for i, v in enumerate(videos):
                print(f"  {i + 1}. {v}")
            try:
                idx = int(input("Escolha o número do vídeo a analisar: ")) - 1
                video_path = videos[idx]
            except (ValueError, IndexError):
                print("❌ Opção inválida.")
                continue

            tempo_str = input("Tempo do frame (ex: 5s, 6500ms; ENTER=último): ").strip()
            frame_custom = None
            if tempo_str:
                try:
                    segundos = parse_tempo_para_segundos(tempo_str)
                    frame_custom = extrair_frame_em_tempo(video_path, segundos)
                except ValueError as e:
                    print(f"⚠️ {e} Usando o último frame.")
                    frame_custom = extrair_ultimo_frame(video_path)
            else:
                frame_custom = extrair_ultimo_frame(video_path)

            analisar_video_e_gerar_prompt(client, cfg, video_path, frame_custom)

        elif escolha == "3":
            if not os.path.exists(cfg.arquivo_analise):
                print(f"❌ Arquivo '{cfg.arquivo_analise}' não encontrado.")
                continue
            conteudo = Path(cfg.arquivo_analise).read_text(encoding="utf-8")
            prompt = extrair_prompt_english(conteudo)
            if not prompt:
                print("❌ Não foi possível extrair o bloco de prompt em inglês.")
                continue

            print(prompt)
            input("Pressione ENTER para gerar o próximo episódio...")
            proximo_num = len(glob.glob("episodio_*.mp4")) + 1
            nome_arquivo = f"episodio_{proximo_num:02d}.mp4"
            gerar_video(client, cfg, prompt, nome_arquivo)

        elif escolha == "4":
            print("👋 Encerrando.")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
