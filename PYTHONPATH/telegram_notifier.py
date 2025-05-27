import asyncio
import telegram
import os
from dotenv import load_dotenv
import time


dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
print(f"[DEBUG telegram_notifier] Mencoba memuat .env dari path: {dotenv_path}")

loaded_successfully = load_dotenv(dotenv_path=dotenv_path)
print(f"[DEBUG telegram_notifier] load_dotenv() berhasil dijalankan: {loaded_successfully}")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


async def send_telegram_message_async(message_text: str):
    """
    Mengirim pesan ke Telegram secara asynchronous.
    """
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_NOTIFIER: Error - TELEGRAM_BOT_TOKEN tidak ditemukan di environment variables atau file .env.")
        return
    if not TELEGRAM_CHAT_ID:
        print("TELEGRAM_NOTIFIER: Error - TELEGRAM_CHAT_ID tidak ditemukan di environment variables atau file .env.")
        return

    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message_text)
        print(f"TELEGRAM_NOTIFIER: Pesan berhasil dikirim.")
    except Exception as e:
        print(f"TELEGRAM_NOTIFIER: Gagal mengirim pesan ke Telegram: {e}")

def send_telegram_notification(message_text: str):
    """
    Wrapper sinkron untuk mengirim pesan Telegram.
    """
    try:
        asyncio.run(send_telegram_message_async(message_text))
    except RuntimeError as e:
        print(f"TELEGRAM_NOTIFIER: Runtime error saat menjalankan asyncio.run: {e}.")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                print("TELEGRAM_NOTIFIER: Event loop sudah berjalan. Mencoba mengirim dengan loop yang ada (mungkin perlu penyesuaian).")
            else:
                asyncio.run(send_telegram_message_async(message_text))
        except Exception as ex_inner:
            print(f"TELEGRAM_NOTIFIER: Gagal mengirim pesan bahkan dengan get_event_loop: {ex_inner}")

if __name__ == '__main__':
    print("Menguji pengiriman pesan Telegram dari PYTHONPATH/telegram_notifier.py...")
    print(f"Token Loaded (dari os.getenv): {TELEGRAM_BOT_TOKEN is not None}")
    print(f"Chat ID Loaded (dari os.getenv): {TELEGRAM_CHAT_ID is not None}") 

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        test_message = f"Ini adalah pesan tes langsung (dengan path .env eksplisit) dari PYTHONPATH/telegram_notifier.py pada {current_time}"
        print(f"Mengirim pesan tes: \"{test_message}\"")
        send_telegram_notification(test_message)
        print("Pesan tes telah dikirim (atau percobaan pengiriman dilakukan).")
    else:
        print("Tidak dapat mengirim pesan tes. Pastikan TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID sudah diatur di file .env Anda.")
        print(f"Path .env yang dicoba: {dotenv_path}")