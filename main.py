import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import flags
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionMiddleware
from loguru import logger

from src.rag_utils import get_chunks, get_embeddings, get_retriever, get_llm
from src.rag import RAG
from src.prompts import RAG_PROMPT
from src import log_handler
from src.config import config


storage = MemoryStorage()
dp = Dispatcher(storage=storage)
dp.message.middleware(ChatActionMiddleware())  # Нужно для анимации набора текста у бота,
                                               # когда происходит генерация ответа

chunks = get_chunks("artifacts/data/lib_data", "*.md")
embeddings = get_embeddings("cointegrated/rubert-tiny2")
retriever = get_retriever(chunks, embeddings)

# IlyaGusev/saiga_llama3_8b, Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24, gpt-3.5-turbo-0125, GigaChat
llm = get_llm(config.llm, use_api=True, api_key=config.gigachat_api_key.get_secret_value())
rag = RAG(llm, retriever, RAG_PROMPT, use_api=True)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    text = f"""
👋 Привет, *{message.from_user.full_name}*! Добро пожаловать в бота библиотеки РУДН!

Я здесь, чтобы помочь вам с любыми вопросами, связанными с библиотекой. Просто напишите мне, и я постараюсь помочь.

Не стесняйтесь обращаться ко мне в любое время! Я здесь, чтобы сделать вашу университетскую жизнь более удобной. 🎓✨
"""

    await message.answer(text, parse_mode="MARKDOWN", disable_web_page_preview=True)


@dp.message()
@flags.chat_action(initial_sleep=0, action="typing", interval=0)
async def cmd_new_chat_handler(message: Message) -> None:
    answer = rag.generate(message.text)
    await message.answer(answer, parse_mode="MARKDOWN", disable_web_page_preview=True)


async def main():
    # Включение логирования
    await log_handler.setup()

    # Инициализация Bot с режимом парсинга HTML (чтобы не было проблем с экранированием)
    bot = Bot(token=config.telegram_bot_token.get_secret_value(), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    bot_user = await bot.me()
    logger.info(f"Initialize Bot: {bot_user.full_name} [@{bot_user.username}]")

    # Удаление всех обновлений, которые пришли, когда бот был неактивен.
    # Это нужно, чтобы он обрабатывал только те сообщения,
    # которые пришли непосредственно во время его работы, а не за всё время.
    logger.info("Drop pendings updates")
    await bot.delete_webhook(drop_pending_updates=True)

    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
