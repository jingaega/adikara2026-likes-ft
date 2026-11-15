# adikara2026-likes-ft

Checkpoint fine-tune untuk notebook ADIKARA 2026 Data Mining (prediksi jumlah likes tweet), tim Suwandi.

## Isi (Releases → v1.0)
- 30 checkpoint `safetensors` (fp16, ±220 MB per file): 6 model × 5 fold.
- 30 file `.txt` kecil berisi skala target (`mu`, `sd`) tiap fold.

| Tag | Input | Seed |
|---|---|---|
| `ft_text`, `ft_text_s2`, `ft_text_s3` | teks tweet | 1, 2, 3 |
| `ft_nbr`, `ft_nbr_s2`, `ft_nbr_s3` | teks tweet + tweet sebelum/sesudah dari user yang sama | 1, 2, 3 |

Nama file: `<tag>_f<fold>.safetensors` dan `<tag>_f<fold>.txt`, fold 0–4.

## Model
- Backbone: [`indolem/indobertweet-base-uncased`](https://huggingface.co/indolem/indobertweet-base-uncased) (Apache-2.0), dilanjutkan dengan domain pretraining (MLM) pada teks tweet kompetisi, lalu fine-tune 5-fold dengan dua head: peluang nol likes dan level `log1p(likes)`.
- `client_device` tidak dipakai.

## Cara pakai
Tidak perlu login atau API key. Notebook mengunduh file dari:

```
https://github.com/jingaega/adikara2026-likes-ft/releases/download/v1.0/<tag>_f<fold>.safetensors
```

Loader (`load_ckpt` di `src/finetune.py` pada notebook) memuat bobot fp16 ke model dan membaca `mu`/`sd` dari file `.txt`.

Repositori ini hanya berisi bobot model, tidak ada data kompetisi.
