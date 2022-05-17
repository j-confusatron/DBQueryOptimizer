import socketserver
from multiprocessing import Process
import configparser
import time
import json
import struct
from query_handler import QueryHandler
from requests import JSONDecodeError

class JsonHandler(socketserver.BaseRequestHandler):
    def setup(self):
        self.__messages = []

    def handle(self):
        str_buf = ''
        while True:
            str_buf += self.request.recv(1024).decode('UTF-8')
            if not str_buf:
                return
            
            if (null_loc := str_buf.find('\n')) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc+1:]
                if json_msg:
                    try:
                        if self.handle_json(json.loads(json_msg)):
                            break
                    except json.decoder.JSONDecodeError:
                        print("Error decoding JSON: ", json_msg)
                        break

    def handle_json(self, data):
        if 'final' in data:
            msg_type = self.__messages[0]['type']
            self.__messages = self.__messages[1:]

            if msg_type == 'query':
                plan = self.server.query_handler.select_plan(self.__messages)
                self.request.sendall(struct.pack("I", plan))
                self.request.close()
            elif msg_type == 'predict':
                result = self.server.query_handler.predict(self.__messages)
                self.request.sendall(struct.pack("I", plan))
                self.request.close()
            elif msg_type == 'reward':
                plan, buffers, obs_reward = self.__messages
                self.server.query_handler.store(plan, buffers, obs_reward)
                pass
            elif msg_type == 'load model':
                path = self.__messages[0]['path']
                self.server.query_handler.load_model(path)
            else:
                print("Unknown message type: ", msg_type)

            return True

        self.__messages.append(data)
        return False

def start_server(addr, port):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((addr, port), JsonHandler) as server:
        server.query_handler = QueryHandler()
        server.serve_forever()
    pass

if __name__ == '__main__':
    # Get server config
    config = configparser.ConfigParser()
    config.read('server.cfg')
    cfg = config['server']
    port = int(cfg['Port'])
    addr = cfg['ListenOn']

    # Startup the server
    server = Process(target=start_server, args=[addr, port])
    server.start()
    print(f"Query Server running @ {addr}:{port}")
    while True:
        time.sleep(60)