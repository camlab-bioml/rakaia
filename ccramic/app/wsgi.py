from ccramic.app.entrypoint import init_app, __version__
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
                        help="automatically open the browser when the app is called. Default: False",
                        dest="auto_open")

    # parser.parse_args(args)

    return parser


def main(sysargs = sys.argv[1:]):

    parser = argparser()
    args = parser.parse_args(sysargs)
    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f'http://127.0.0.1:5000/')

    app = init_app()
    port = 5000
    if args.auto_open:
        Timer(1, open_browser).start()
    app.run(host='0.0.0.0', debug=True, threaded=True, port=port)

if __name__ == "__main__":
    main()
