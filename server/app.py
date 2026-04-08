from src.envs.my_env.server.app import app as env_app
import uvicorn

app = env_app

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()