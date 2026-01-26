import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Configurações
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

async def reuniao_marketing():
    model_client = OpenAIChatCompletionClient(model=model_name, api_key=api_key)

    # 2. O Estrategista (IA)
    estrategista = AssistantAgent(
        name="Estrategista",
        model_client=model_client,
        system_message="Você define os pilares de conteúdo. Foque em autoridade e educação sobre tecnologia no esporte."
    )

    # 3. O Copywriter (IA)
    copywriter = AssistantAgent(
        name="Copywriter",
        model_client=model_client,
        system_message="Você transforma os pilares em legendas criativas e chamativas para Instagram e LinkedIn."
    )

    # 4. VOCÊ (Humano)
    # O UserProxyAgent vai parar a execução e pedir seu input no terminal
    voce = UserProxyAgent(name="Diretor_Criativo")

    # 5. Condição de Parada
    termination_condition = TextMentionTermination("APROVADO")

    # 6. O Time de Marketing
    # Ordem: Estrategista -> Copywriter -> Você avalia
    time_marketing = RoundRobinGroupChat(
        [estrategista, copywriter, voce],
        termination_condition=termination_condition
    )

    tarefa_inicial = "elabore uma linha editorial para arquitetura "

    print(f"📣 REUNIÃO DE MARKETING INICIADA")
    print(f"Dica: Digite 'APROVADO' para encerrar ou dê suas instruções para os agentes.\n")

    # Execução
    async for message in time_marketing.run_stream(task=tarefa_inicial):
        if hasattr(message, 'content'):
            print(f"\n📢 [{message.source.upper()}]:")
            print(f"{message.content}")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(reuniao_marketing())
