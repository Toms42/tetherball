import argparse
import random
import time

from pythonosc import udp_client

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip", default="172.26.162.185",
      help="The ip of the OSC server")
  parser.add_argument("--port", type=int, default=10000,
      help="The port the OSC server is listening on")
  args = parser.parse_args()

  client = udp_client.SimpleUDPClient(args.ip, args.port)

  for x in range(60):
    print("hey tom")
    client.send_message("/hey", random.random())
    client.send_message("/tom", random.random())
    time.sleep(1)