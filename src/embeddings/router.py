import base64
import tempfile
from datauri import DataURI
from fastapi import APIRouter
from src.embeddings import service as embedding_service

router = APIRouter()


@router.post("/embeddings")
async def embeddings():
    _input: str = ""

    uri = DataURI(_input)
    if uri.mimetype != 'audio/mpeg':
        raise Exception("Not supported!")

    # TODO write into tmp file
    uri.data


    with urlopen(_input) as response:
        mimetype = response.info().get_content_type()

        data = response.read()
        image = base64.b64encode(data)

    # 1. 解碼 Base64 資料
    header, data = (
        audio_base64.split(",") if "," in audio_base64 else (None, audio_base64)
    )
    audio_bytes = base64.b64decode(data)

    # 2. 建立暫時檔案 (delete=True 會在檔案關閉時自動刪除)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()  # 確保資料已寫入磁碟

        # 3. 取得路徑並傳給函式庫
        result = process_audio(temp_file.name)

        return {"status": "success", "result": result}
