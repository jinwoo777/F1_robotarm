"""테이블별 QR코드 이미지 생성.

각 테이블에 붙일 QR코드를 미리 만들어서 qrcodes/ 폴더에 저장한다. QR을 찍으면
서버의 "/?table=N" 으로 접속되고, jumak_order.html이 그 번호를 화면에 표시하며
주문 시 함께 저장한다 (app.py의 index(), database.py의 table_number 참고).

사용법: python3 generate_table_qr.py
"""
import socket
import qrcode

PORT = 5000
TABLE_COUNT = 1  # 매장 테이블(홀) 개수 — 늘어나면 이 숫자만 바꾸고 다시 실행
OUTPUT_DIR = "qrcodes"


def detect_lan_ip():
    """이 PC가 매장 와이파이에서 쓰는 실제 IP를 알아낸다(패킷을 보내지는 않음)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ip = detect_lan_ip()
    print(f"감지된 서버 IP: {ip} (공유기에서 이 PC에 고정 IP를 할당해뒀는지 확인하세요)")

    for table_no in range(1, TABLE_COUNT + 1):
        url = f"http://{ip}:{PORT}/?table={table_no}"
        img = qrcode.make(url)
        path = os.path.join(OUTPUT_DIR, f"table_{table_no:02d}.png")
        img.save(path)
        print(f"홀 {table_no:02d} -> {url}  저장: {path}")


if __name__ == "__main__":
    main()
