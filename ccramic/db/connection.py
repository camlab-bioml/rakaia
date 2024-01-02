from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from ccramic.utils.db import format_blend_config_document_for_insert

class AtlasDatabaseConnection:
    """
    This class represents a user connection to a mongoDB Atlas database
    """
    def __init__(self, username: str = None, password: str = None, database_name: str = "ccramic",
                 blend_collection_name: str = "blend_config"):
        self.username = username
        self.password = password
        self.connection_string = f"mongodb+srv://{self.username}:{self.password}@ccramic-db" \
        f".uzqznla.mongodb.net/?retryWrites=true&w=majority"
        self.client = MongoClient(self.connection_string, server_api=ServerApi('1'))
        # set the name of the database and collection
        self.database = self.client[database_name]
        self.blend_collection = self.database[blend_collection_name]

    def ping_connection(self):
        try:
            self.client.admin.command('ping')
            return True, "Connection to ccramic-db successful"
        except Exception as e:
            return False, f"Connection to ccramic-db failed: \n {e}"

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
                            global_filter_type, global_filter_val, global_filter_sigma, alias_dict=None):
        """
        Insert a blend config document into the `blend_config` collection.
        Important: will overwrite any previous configs from the user with the same name
        """
        # delete any configs that match the name provided (overwrite)
        delete = self.blend_collection.delete_many({"user": self.username, "name": config_name})
        insert = self.blend_collection.insert_one(format_blend_config_document_for_insert(
            self.username, config_name, blend_dict, selected_channel_list, global_apply_filter,
                                    global_filter_type, global_filter_val, global_filter_sigma, alias_dict))
    def username_password_pair(self):
        return {'username': self.username, 'password': self.password}

    def remove_blend_document_by_name(self, config_name):
        """
        Remove a document by the `name` key of the document
        """
        delete = self.blend_collection.delete_many({"user": self.username, "name": config_name})

    def close(self):
        self.client.close()
