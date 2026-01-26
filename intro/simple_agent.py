import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 🔹 Carregar variáveis do .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # Recomendado gpt-4o-mini para custo

async def chat_loop():
    # 🔹 CONFIGURAÇÃO CORRIGIDA
    # Em muitas versões do autogen-ext, passamos os parâmetros de geração aqui:
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        # As configurações de performance entram aqui como argumentos diretos
        max_tokens=1024,
        temperature=0.2,
    )

    # 🔹 Criar o agente
    agent = AssistantAgent(
        name="chatbot",
        model_client=model_client,
        system_message="Você é um assistente útil para o projeto Nexos. Seja direto e conciso."
    )

    print(f"🤖 Chatbot Nexos ({model_name}) iniciado! Digite 'sair' para encerrar.")

    while True:
        # Usamos run_in_executor para o input não travar o loop async se necessário,
        # mas para este script simples, o input direto funciona.
        user_input = input("\nVocê: ")
        if user_input.lower() in ("sair", "exit", "quit"):
            print("👋 Encerrando o chat...")
            break

        try:
            input_message = TextMessage(content=user_input, source="user")

            # Enviando a mensagem para o agente
            response = await agent.on_messages(
                [input_message],
                cancellation_token=None
            )

            print(f"Assistente: {response.chat_message.content}")

        except Exception as e:
            # Se o erro persistir nos argumentos, mostramos o detalhe aqui
            print(f"[ERRO] Ocorreu um problema: {e}")

if __name__ == "__main__":
    asyncio.run(chat_loop())