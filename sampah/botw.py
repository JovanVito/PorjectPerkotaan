from telegram.ext import Application, CommandHandler, MessageHandler, filters # Perhatikan 'filters' huruf kecil

# !!! GANTI DENGAN TOKEN BARU ANDA SETELAH REVOKE !!!
TOKEN = "7905613614:AAE24pVdV5GgFGQBYdtiHOgWEY_ggxWaJUE"

async def start(update, context): # Fungsi handler sekarang biasanya async
    """Kirim pesan ketika perintah /start dikeluarkan."""
    await update.message.reply_text('Halo! Saya bot baru.')

async def echo(update, context): # Fungsi handler sekarang biasanya async
    """Mengulang pesan yang dikirim pengguna."""
    await update.message.reply_text(update.message.text)

def main():
    """Mulai bot."""
    # Buat Application instance
    application = Application.builder().token(TOKEN).build()

    # Tambahkan handler untuk berbagai perintah dan pesan
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo)) # Gunakan filters.TEXT dan filters.COMMAND

    # Jalankan bot sampai pengguna menekan Ctrl-C
    print("Bot sedang berjalan. Tekan Ctrl-C untuk menghentikan.")
    application.run_polling()

if __name__ == '__main__':
    main()