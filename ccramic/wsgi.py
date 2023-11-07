from ccramic.entrypoint import init_app, __version__
import argparse
import sys
import webbrowser
from threading import Timer
import os

def argparser():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="ccramic: Cell-type Classification (using) Rapid Analysis (of) Multiplexed "
                                                 "Imaging (mass) Cytometry using Flask and Dash.",
                                     usage='''
            ccramic can be initialized from the command line using: \n
            ccramic \n
            From here, navigate to http://127.0.0.1:5000/ or http://0.0.0.0:5000/ to access ccramic.''')

    parser.add_argument('-v', "--version", action="version",
                        help="Show the current ccramic version then exit.",
                        version=f"This is ccramic: v{__version__}")

    parser.add_argument('-h', "--help", action="help",
                        help="Show the help output and exit.",
                        dest="help")
    parser.add_argument('-a', "--auto-open", action="store_true",
                        help="automatically open the browser when the dash is called. Default: False",
                        dest="auto_open")
    parser.add_argument('-l', "--use-local-dialog", action="store_true",
                        help="Enable a local file dialog with wxPython to browse and read local files. Default: False",
                        dest="use_local_dialog")
    parser.add_argument('-p', "--port", action="store",
                        help="Set the port for ccramic on local runs. Default: 5000. Other options to consider are "
                             "8050, 8080",
                        dest="port", default=5000, type=int)

    return parser


def main(sysargs = sys.argv[1:]):

    parser = argparser()
    args = parser.parse_args(sysargs)
    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f'http://127.0.0.1:{args.port}/')

    # establish the cli config

    CLI_CONFIG = {"use_local_dialog": args.use_local_dialog}

    app = init_app(cli_config=CLI_CONFIG)
    if args.auto_open:
        Timer(1, open_browser).start()
    app.run(host='0.0.0.0', debug=True, threaded=True, port=args.port)

if __name__ == "__main__":
    main()
