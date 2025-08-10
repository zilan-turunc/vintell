# Otomasyon ve Agent Mantığı: Moodboard Etiket Önerici Agent

## Amaç:
Kullanıcının verdiği kıyafet ya da tarz açıklamasına göre, sosyal medya için uygun 5 etiket önerilir. Bu işlem bir agent tarafından otomatik yapılır.

## Teknoloji:
- OpenAI GPT-3.5 Turbo
- Python

## Çalışma Şekli:
1. Kullanıcı bir moda açıklaması girer (örneğin: "kırmızı bluz").
2. Agent bu açıklamayı OpenAI API'ye gönderir.
3. OpenAI cevaben 5 kısa etiket önerir (örn: #vintage, #pinterest, #crop).
4. Bu etiketler terminale yazdırılır.

## Çalıştırmak için:
1. `pip install openai`
2. `moodboard_agent.py` dosyasını aç ve kendi OpenAI API anahtarını gir.
3. Terminalde çalıştır:
```bash
python agents/moodboard_agent.py
