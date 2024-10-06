import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import flags
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionMiddleware
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from src import log_handler
from src.graph.graph import app
from src.graph.consts import WELCOME_MESSAGE, SORRY_MESSAGE
from src.config import config


storage = MemoryStorage()
dp = Dispatcher(storage=storage)
dp.message.middleware(ChatActionMiddleware())  # Нужно для анимации набора текста у бота,
                                               # когда происходит генерация ответа
user_history = {}
# # IlyaGusev/saiga_llama3_8b, Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24, gpt-3.5-turbo-0125, GigaChat


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    text = WELCOME_MESSAGE.format(full_name=message.from_user.full_name)

    await message.answer(text, parse_mode="MARKDOWN", disable_web_page_preview=True)


@dp.message()
@flags.chat_action(initial_sleep=0, action="typing", interval=0)
async def cmd_message_handler(message: Message) -> None:
    try:
        # Запуск анимации печатания
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

        user_id = message.from_user.id
        question = message.text

        # Создаем историю чата, если такой еще нет
        if user_id not in user_history:
            user_history[user_id] = ChatMessageHistory(max_messages=30)

        # Добавляем в историю сообщение пользователя
        user_history[user_id].add_message(HumanMessage(content=question))

        # Передаем состояние в граф и получаем ответ
        answer = app.invoke(input={
            "question": question,
            "chat_history": await user_history[user_id].aget_messages()
        })["generation"]

        # Сохраняем ответ в историю
        user_history[user_id].add_message(AIMessage(content=answer))

        await message.answer(answer, parse_mode="MARKDOWN", disable_web_page_preview=True)

    except Exception as e:
        error_message = SORRY_MESSAGE
        await message.answer(error_message)
        # Можно также добавить логирование ошибки
        logger.error(f"An error occurred: {str(e)}")


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
