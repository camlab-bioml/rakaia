from ccramic.app import init_app


def main():
    app = init_app()
    app.run(host='0.0.0.0', debug=True)


if __name__ == "__main__":
    main()