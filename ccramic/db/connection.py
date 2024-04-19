from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from ccramic.utils.db import format_blend_config_document_for_insert
from pymongo.collection import Collection
from pymongo.errors import ConfigurationError

class AtlasDatabaseConnection:
    """
    This class represents a user connection to a mongoDB Atlas database
    An existing client can be passed, or a new one can be created using a connection string and username/password
    combination. By default, a collection is created using the sandbox connection (for now)
    The client should have the following configurations: a database corresponding to the `database_name` i.e. `ccramic`,
    and inside the database, a collection corresponding to the `blend_collection_name` i.e. `blend_config`
    """
    def __init__(self, connection_string: str="ccramic-db.uzqznla.mongodb.net",
                 username: str = None, password: str = None, database_name: str = "ccramic",
                 blend_collection_name: str = "blend_config", existing_client: MongoClient=None):
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

    def create_connection(self, new_collection: Collection = None):
        try:
            self.client = MongoClient(self.connection_string, server_api=ServerApi('1')) if not self.existing_client \
                else self.existing_client
            # set the name of the database and collection
            self.database = self.client[self.database_name]
            self.blend_collection = self.database[self.blend_collection_name] if not \
                new_collection else new_collection
            self.client.admin.command('ping')
            return True, f"Connection to database: {self.database.name} successful"
        except (AttributeError, ConfigurationError, Exception) as e:
            return False, f"Connection to database: {self.database.name} failed: \n {e}"

    def blend_configs_by_user(self, user_key="user", id_key="name"):
        """
        Returns a tuple: first element is a list of blend config dictionaries, and the second is the list of the names
        blend config names are stored as `name` in the document
        """
        blend_names = []
        configs = []
        query = self.blend_collection.find({user_key: self.username})
        for result in query:
            configs.append(result)
            blend_names.append(str(result[id_key]))
        return configs, blend_names

    def insert_blend_config(self, config_name, blend_dict, selected_channel_list, global_apply_filter,
                            global_filter_type, global_filter_val, global_filter_sigma, data_selection: str=None,
                            cluster_assignments: dict=None, alias_dict: dict=None, gating_dict: dict=None):
        """
        Insert a blend config document into the `blend_config` collection.
        Important: will overwrite any previous configs from the user with the same name
        """
        # delete any configs that match the name provided (overwrite)
        delete = self.blend_collection.delete_many({"user": self.username, "name": config_name})
        insert = self.blend_collection.insert_one(format_blend_config_document_for_insert(
            self.username, config_name, blend_dict, selected_channel_list, global_apply_filter,
                                    global_filter_type, global_filter_val, global_filter_sigma, data_selection,
                                    cluster_assignments, alias_dict, gating_dict))
    def username_password_pair(self):
        return {'username': self.username, 'password': self.password}

    def remove_blend_document_by_name(self, config_name):
        """
        Remove a document by the `name` key of the document
        """
        delete = self.blend_collection.delete_many({"user": self.username, "name": config_name})

    def close(self):
        if self.client is not None:
            self.client.close()
