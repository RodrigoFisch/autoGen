import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage  # Importação necessária
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 🔹 Carregar variáveis do .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")


async def chat_loop():
    # 🔹 Configurar o cliente corretamente
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key
    )

    # 🔹 Criar o agente
    agent = AssistantAgent(
        name="chatbot",
        model_client=model_client,
        system_message="Você é um assistente útil para o projeto Nexos."
    )

    print("🤖 Chatbot Nexos iniciado! Digite 'sair' para encerrar.")

    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ("sair", "exit", "quit"):
            print("👋 Encerrando o chat...")
            break

        try:
            # ✅ CORREÇÃO: Usar TextMessage em vez de um dicionário dict
            input_message = TextMessage(content=user_input, source="user")

            # Enviando a mensagem para o agente
            response = await agent.on_messages(
                [input_message],
                cancellation_token=None
            )

            # ✅ CORREÇÃO: Acessar o conteúdo da mensagem de resposta
            # O objeto response contém a propriedade chat_message
            print(f"Assistente: {response.chat_message.content}")

        except Exception as e:
            print(f"[ERRO] Ocorreu um problema: {e}")


if __name__ == "__main__":
    asyncio.run(chat_loop())