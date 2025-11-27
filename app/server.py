from flask import Flask, request, jsonify
from flask_sock import Sock
import logging
import os
import asyncio
from Utils import *
from flask_cors import CORS
import threading



class Server:

    def __init__(self):
        self.app = Flask(__name__)

        # To avoid CORS errors
        CORS(self.app)
        self.sock = Sock(self.app)
        self.connected_clients = set()

        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/start', 'start_processing', self.start_processing, methods=['POST'])

        self.sock.route('/stream')(self.stream)

    def send_chunk(self, text: str):
        clients = list(self.connected_clients)
        for ws in clients:
            try:
                ws.send(text)
            except Exception as e:
                self.connected_clients.discard(ws)

    # To manage incoming WebSocket connections
    def stream(self, ws):
        self.connected_clients.add(ws)
        try:
            while True:
                _ = ws.receive()
        except Exception:
            print(f"Client disconnected: {ws}")
        finally:
            self.connected_clients.discard(ws)


    # Processing after POST request
    def start_processing(self):
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        text = data["text"]

        load_env_variables()
        log_path = os.getenv("LOG_PATH")
        set_logging_config(log_path)
        llm = get_config_data("../config.yaml")

        thread = threading.Thread(target=self.run_main_in_background, args=(llm, text))
        thread.start()

        return jsonify({"status": "ok", "message": "Processing started in background", "received": text})

    def run_main_in_background(self, llm, user_text):
        from main import main
        try:
            # Crea un nuovo event loop per il thread corrente
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main(llm, self))
        except Exception as e:
            print(f"Error during background execution: {e}")
        finally:
            logging.shutdown()

    # Start server Flask
    def run(self, host='127.0.0.1', port=5000, debug=True, use_reloader=False):
        self.app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)


if __name__ == "__main__":
    server = Server()
    server.run()