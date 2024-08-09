from typing import Union
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection
from pymongo.errors import ConfigurationError
from rakaia.utils.db import format_blend_config_document_for_insert


class AtlasDatabaseConnection:
    """
    Provides a user connection to a mongoDB Atlas database
    An existing client can be passed, or a new one can be created using a connection string and username/password
    combination. By default, a collection is created using the sandbox connection (for now)
    The client should have the following configurations: a database corresponding to the `database_name` i.e. `rakaia`,
    and inside the database, a collection corresponding to the `blend_collection_name` i.e. `blend_config`

    :param connection_string: MongoDB/Atlas compatible connection string to a cloud database
    :param username: Username
    :param password: User password
    :param database_name: Name of the database to connect to in the project specified by `connection_string`
    :param blend_collection_name: string of the collection in the `database_name`. Default is `blend_config`
    :param existing_client: Pass an existing open client to overwrite the previous credentials
    :return: None
    """

    def __init__(self, connection_string: str = "rakaia-db.uzqznla.mongodb.net",
                 username: str = None, password: str = None, database_name: str = "rakaia",
                 blend_collection_name: str = "blend_config", existing_client: MongoClient = None) -> None:
        self.username = username
        self.password = password
        self.connection_string = f"mongodb+srv://{self.username}:{self.password}@" \
                                 f"{connection_string}/?retryWrites=true&w=majority"
        self.database_name = database_name
        self.blend_collection_name = blend_collection_name
        self.client = None
        self.database = None
        self.blend_collection = None
        self.existing_client = existing_client

    def create_connection(self, new_collection: Collection = None) -> tuple:
        """
        Create a user connection to the database.

        :param new_collection: String for a new collection to create.
        :return: tuple: Boolean indicating successful connection, and a string user alert representing the boolean
        """
        try:
            self.client = MongoClient(self.connection_string, server_api=ServerApi('1')) if not self.existing_client \
                else self.existing_client
            # set the name of the database and collection
            self.database = self.client[self.database_name]
            self.blend_collection = self.database[self.blend_collection_name] if not \
                new_collection else new_collection
            self.client.admin.command('ping')
            return True, f"Connection to database: {self.database.name} successful"
        except (AttributeError, ConfigurationError, Exception) as except_db:
            return False, f"Connection to database: {self.database.name} failed: \n {except_db}"

    def blend_configs_by_user(self, user_key="user", id_key="name") -> tuple:
        """
        Get all blend configs associated with a user in the shared database.

        :param user_key: Username to search on
        :param id_key: Document key that stores the value of the blend name set by the user. Default is `name`
        :return: List of blend config dictionaries, and list of blend config names associated with one user
        """
        blend_names = []
        configs = []
        query = self.blend_collection.find({user_key: self.username})
        for result in query:
            configs.append(result)
            blend_names.append(str(result[id_key]))
        return configs, blend_names

    def insert_blend_config(self, config_name, blend_dict, selected_channel_list, global_apply_filter,
                            global_filter_type, global_filter_val, global_filter_sigma, data_selection: str = None,
                            cluster_assignments: dict = None, alias_dict: dict = None, gating_dict: dict = None,
                            mask_toggle: bool = False, mask_level: Union[int, float] = 35, mask_boundary: bool = True,
                            mask_hover: Union[bool, list] = False):
        """
        Insert a blend config document into the `blend_config` collection.
        Important: will overwrite any previous configs from the user with the same name

        :param config_name: User set name of the blend config
        :param blend_dict: Dictionary of current channel blend parameters
        :param selected_channel_list: List of channels in the current blend
        :param global_apply_filter: Whether or not a global filter has been applied
        :param global_filter_type: String specifying a global gaussian or median blur
        :param global_filter_val: Kernel size for the global gaussian or median blur
        :param global_filter_sigma: If global gaussian blur is applied, set the sigma value
        :param data_selection: String representation of the current session ROI selection
        :param cluster_assignments: Dictionary of mask cluster categories matching to a cluster color
        :param alias_dict: Dictionary matching channel keys to their displayed labels
        :param gating_dict: Dictionary of current gating parameters
        :param mask_toggle: Whether to show the mask over the channel blend
        :param mask_level: Set opacity of mask relative to the blend image. Takes a value between 0 and 100
        :param mask_boundary: Whether to include the object boundaries in the mask projection
        :param mask_hover: Whether to include the mask object ID in the hover template
        :return: None
        """
        # delete any configs that match the name provided (overwrite)
        self.blend_collection.delete_many({"user": self.username, "name": config_name})
        self.blend_collection.insert_one(format_blend_config_document_for_insert(
            self.username, config_name, blend_dict, selected_channel_list, global_apply_filter,
            global_filter_type, global_filter_val, global_filter_sigma, data_selection,
            cluster_assignments, alias_dict, gating_dict,
            mask_toggle, mask_level, mask_boundary, mask_hover))

    def username_password_pair(self) -> dict:
        """
        Get the current connection username and password

        :return: Dictionary containing the username and password
        """
        return {'username': self.username, 'password': self.password}

    def remove_blend_document_by_name(self, config_name):
        """
        Remove a document by the `name` key of the document
        """
        self.blend_collection.delete_many({"user": self.username, "name": config_name})

    def close(self):
        """
        Close the mongoDB Atlas connection for the current client.

        :return: None
        """
        if self.client is not None:
            self.client.close()
