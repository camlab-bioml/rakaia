import tempfile
import argparse
import sys
import webbrowser
import os
from threading import Timer
from waitress import serve
from rakaia.entrypoint import init_app, __version__


def cli_parser():
    """Retrieve the argparse parser arguments on aoo CLI initialization.

    :return: argparse Argument parser with provided CLI options
    """
    parser = argparse.ArgumentParser(add_help=False,
                                     description="rakaia: Analyze multiplexed imaging datasets interactively and quickly.",
                                     usage='''
            rakaia can be initialized from the command line using: \n
            rakaia \n
            From here, navigate to http://127.0.0.1:5000/ or http://0.0.0.0:5000/ to access rakaia. \n
            If a different port is used, replace 5000 with the provided port number.''')

    parser.add_argument('-v', "--version", action="version",
                        help="Show the current rakaia version then exit. Does not execute the application.",
                        version=f"This is rakaia: v{__version__}")

    parser.add_argument('-h', "--help", action="help",
                        help="Show the help/options menu and exit. Does not execute the application.",
                        dest="help")
    parser.add_argument('-a', "--auto-open", action="store_true",
                        help="automatically open the browser when the dash is called. Default: False",
                        dest="auto_open")
    parser.add_argument('-l', "--use-local-dialog", action="store_true",
                        help="Enable a local file dialog with wxPython to browse and read local files. Default: False",
                        dest="use_local_dialog")
    parser.add_argument('-p', "--port", action="store",
                        help="Set the port for rakaia on local runs. Default: 5000. Other options to consider are "
                             "8050, 8080",
                        dest="port", default=5000, type=int)
    parser.add_argument('-pr', "--production-mode", action="store_false",
                        help="Enable production mode. This will generate a production-level WSGI server,"
                             "and will switch the application out of debug mode. Default: not enabled",
                        dest="is_dev_mode")
    parser.add_argument('-dt', "--disable-threading", action="store_false",
                        help="Disable threading. By default, threading is enabled.",
                        dest="threading")
    parser.add_argument('-dl', "--disable-loading", action="store_false",
                        help="Disable loading on data import and data switching. By default, loading is enabled.",
                        dest="loading")
    parser.add_argument('-dp', "--disable-persistence", action="store_false",
                        help="Disable saving persistent session variable values in the browser. By default, persistence is enabled.",
                        dest="persistence")
    parser.add_argument('-dc', "--disable-cache-overwriting", action="store_false",
                        help="Disable cache overwriting for server side stores. By default, rakaia will overwrite `Serverside` objects\n"
                             "on each callback invocation to save disk space. However, overwriting should be disabled for concurrent sessions\n"
                             "that host multiple users, such as containerized or public instances.",
                        dest="serverside_overwrite")
    parser.add_argument('-sc', "--swatch-colors", action="store",
                        help="Set custom RGB codes for the swatches. Should be a string wrapped in quotations of the form \n"
                             "`#FF0000,#00FF00,#0000FF,#00FAFF,#FF00FF,#FFFF00,#FFFFFF`",
                        dest="swatches", default=None, type=str)
    parser.add_argument('-at', "--array-type", action="store",
                        help="Set the preferred numpy array type for storing arrays in session. Options are `float` for "
                             "np.float32, or `int` for np.uint16. Float arrays will have more precision and may "
                             "be required for array values between 0 and 1, but uint arrays will likely consume less memory. "
                             "Default is float",
                        dest="array_type", default="float", type=str, choices=["float", "int"])
    parser.add_argument('-cs', "--cache-storage", action="store",
                        help="Set the output path for the temporary session caches. Default is the temp directory",
                        dest="cache_dest", default=str(tempfile.gettempdir()), type=str)
    parser.add_argument('-tr', "--threads", action="store",
                        help="Set the number of threads to use in production mode. Default is 8",
                        dest="threads", default=8, type=int)

    return parser


def main(sysargs=sys.argv[1:]):
    """
    Define the main entry point for the parent flask app
    Takes the CLI arguments and creates a Flask app instance (dev or through waitress if production mode is used).

    :return: None
    """
    parser = cli_parser()
    args = parser.parse_args(sysargs)

    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f'http://127.0.0.1:{args.port}/')

    # establish the cli config

    CLI_CONFIG = {"use_local_dialog": args.use_local_dialog,
                  'use_loading': args.loading,
                  'persistence': args.persistence,
                  'swatches': args.swatches,
                  'array_store_type': args.array_type,
                  'serverside_overwrite': args.serverside_overwrite,
                  'is_dev_mode': args.is_dev_mode,
                  'cache_dest': args.cache_dest}

    app = init_app(cli_config=CLI_CONFIG)
    #   https://stackoverflow.com/questions/64107108/what-is-the-issue-with-binding-to-all-interfaces-and-what-are-the-alternatives
    HOST = '127.0.0.1' if CLI_CONFIG['is_dev_mode'] else '0.0.0.0'
    if args.auto_open:
        Timer(1, open_browser).start()

    if CLI_CONFIG['is_dev_mode']:
        app.run(host=HOST, debug=args.is_dev_mode, threaded=args.threading, port=args.port)
    else:
        serve(app, host=HOST, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
