import uvicorn


def main() -> None:
    uvicorn.run("evidence_agent.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
