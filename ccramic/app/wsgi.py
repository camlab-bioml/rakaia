from ccramic.app import init_app


def main():
    app = init_app()
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5000)


if __name__ == "__main__":
    main()