from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


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
        Return a tuple: first element is a list of blend config dictionaries, and the second is the list of the names
        blend config names are stored as `name` in the document
        """
        blend_names = []
        configs = []
        query = self.blend_collection.find({user_key: self.username})
        for result in query:
            configs.append(result)
            blend_names.append(str(result[id_key]))
        return configs, blend_names

    def insert_blend_config(self, channel_config, selected_channel_list, global_filter_dict):
        """
        Insert a blend config document into the `blend_config` collection
        """
        raise NotImplementedError

    def username_password_pair(self):
        return {'username': self.username, 'password': self.password}
