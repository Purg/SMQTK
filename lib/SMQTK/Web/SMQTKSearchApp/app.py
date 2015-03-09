"""
SearchApp application object
"""

import flask
import logging
import os.path

from SMQTK.utils.MongoSessions import MongoSessionInterface
from SMQTK.utils import DatabaseInfo


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SMQTKSearchApp (flask.Flask):

    ENV_CONFIG = "SMQTK_SEARCHAPP_CONFIG"

    @property
    def log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    def __init__(self, config_filepath=None):
        super(SMQTKSearchApp, self).__init__(
            self.__class__.__name__,
            static_folder=os.path.join(SCRIPT_DIR, 'static'),
            template_folder=os.path.join(SCRIPT_DIR, 'templates')
        )

        #
        # Configuration setup
        #
        config_env_loaded = config_file_loaded = None

        # Load default -- This should always be present, aka base defaults
        self.config.from_object('smqtk_config')
        config_default_loaded = True

        # Load from env var if present
        if self.ENV_CONFIG in os.environ:
            self.log.info("Loading config from env var (%s)...",
                          self.ENV_CONFIG)
            self.config.from_envvar(self.ENV_CONFIG)
            config_env_loaded = True

        # Load from configuration file if given
        if config_filepath and os.path.isfile(config_filepath):
            config_file_path = os.path.expanduser(os.path.abspath(config_filepath))
            self.log.info("Loading config from file (%s)...", config_file_path)
            self.config.from_pyfile(config_file_path)
            config_file_loaded = True

        self.log.debug("Config defaults loaded : %s", config_default_loaded)
        self.log.debug("Config from env loaded : %s", config_env_loaded)
        self.log.debug("Config from file loaded: %s", config_file_loaded)
        if not (config_default_loaded or config_env_loaded or config_file_loaded):
            raise RuntimeError("No configuration file specified for loading. "
                               "(%s=%s) (file=%s)"
                               % (self.ENV_CONFIG,
                                  os.environ.get(self.ENV_CONFIG, None),
                                  config_filepath))

        self.log.debug("Configuration loaded: %s", self.config)

        #
        # Security
        #
        self.secret_key = self.config['SECRET_KEY']

        #
        # Database setup using Mongo
        #
        h, p = self.config['MONGO_SERVER'].split(':')
        n = "SMQTK_SearchApp"
        self._db_info = DatabaseInfo(h, p, n)

        # Use mongo for session storage.
        # -> This allows session modification during AJAX routines (default
        #    Flask sessions do not)
        self.session_interface = MongoSessionInterface(self._db_info.host,
                                                       self._db_info.port,
                                                       self._db_info.name)

        #
        # Misc. Setup
        #

        # Add 'do' statement usage
        self.jinja_env.add_extension('jinja2.ext.do')

        #
        # Modules
        #
        # Load up required and optional module blueprints
        #

        self.log.debug("Importing Login module")
        from SMQTK.Web.common_flask_blueprints.login import LoginMod
        self.module_login = LoginMod(self)
        self.register_blueprint(self.module_login)

        # self.log.debug("Importing Upload module")
        # self.upload_working_dir = os.path.join(self.config['WORK_DIR'],
        #                                        'UploadWork')
        # from SMQTK.Web.common_flask_blueprints.file_upload import FileUploadMod
        # self.module_upload = FileUploadMod(self, self.upload_working_dir)
        # self.register_blueprint(self.module_upload,
        #                         url_prefix="/upload")

        self._log.debug("Importing Video Search module")
        from .modules.vsearch
        self.module_search = SearchMod(self)
        self.register_blueprint(self.module_search,
                                url_prefix="/search")

        #
        # Basic routing
        #

        @self.route('/home')
        @self.route('/')
        def smqtk_index():
            self.log.info("Session: %s", flask.session.items())
            return flask.render_template("index.html")

    def run(self, host=None, port=None, debug=False, **options):
        """
        Override of the run method, drawing running host and port from
        configuration by default. 'host' and 'port' values specified as argument
        or keyword will override the app configuration.
        """
        super(SMQTKSearchApp, self).run(host=(host or self.config['RUN_HOST']),
                                        port=(port or self.config['RUN_PORT']),
                                        **options)
