import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Configurações
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

async def debate_com_critico():
    model_client = OpenAIChatCompletionClient(model=model_name, api_key=api_key)

    # 2. O Analista (Dados)
    analista = AssistantAgent(
        name="Analista_Dados",
        model_client=model_client,
        system_message="Você analisa estatísticas. Foque em métricas de GPS e rendimento físico. Seja direto."
    )

    # 3. O Preparador (Saúde)
    preparador = AssistantAgent(
        name="Preparador_Fisico",
        model_client=model_client,
        system_message="Você foca em fisiologia e treinos. Proponha soluções práticas de recuperação ou carga de trabalho."
    )

    # 4. O Crítico (Diretor Técnico)
    critico = AssistantAgent(
        name="Diretor_Tecnico",
        model_client=model_client,
        system_message="""Você é o Diretor Técnico do Nexos. 
        Sua função é avaliar se o que o Analista e o Preparador propuseram é viável e estratégico. 
        Se a solução estiver completa e aprovada, encerre a conversa escrevendo a palavra: FINALIZADO. 
        Se não estiver boa, aponte as falhas e peça para eles revisarem."""
    )

    # 5. Condição de Parada
    termination_condition = TextMentionTermination("FINALIZADO")

    # 6. O Time (A ordem será: Analista -> Preparador -> Diretor)
    time_nexos = RoundRobinGroupChat(
        [analista, preparador, critico],
        termination_condition=termination_condition
    )

    # Tarefa inicial
    tarefa = "O lateral-direito está apresentando alto risco de lesão muscular após a sequência de 5 jogos. Como proceder?"

    print(f"🏟️  REUNIÃO TÉCNICA NEXOS INICIADA\n")

    # Execução
    async for message in time_nexos.run_stream(task=tarefa):
        # Filtra para imprimir apenas as mensagens de texto dos agentes
        if hasattr(message, 'content') and message.source != "user":
            print(f"--- {message.source.upper()} ---")
            print(f"{message.content}\n")
            print("-" * 30)

if __name__ == "__main__":
    asyncio.run(debate_com_critico())