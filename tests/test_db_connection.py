import pytest
from rakaia.db.connection import AtlasDatabaseConnection
import pymongo
from pymongo.errors import OperationFailure, InvalidOperation, ServerSelectionTimeoutError
from conftest import skip_on
import mongomock

@skip_on(ServerSelectionTimeoutError, "A connection to the mongoDB server could not be established")
def test_real_db_connection():
    client = mongomock.MongoClient()
    database = client.__getattr__('rakaia')
    collection = database.__getattr__('blend_config')
    atlas_conn = AtlasDatabaseConnection(username="test_user", password="None", existing_client=client)
    atlas_conn.create_connection()
    assert atlas_conn.blend_configs_by_user(user_key="test_user") == ([], [])
    blend_dict = {"ArAr80": {"color": "#FFFFFF", "x_lower_bound": None, "x_upper_bound": None, "filter_type": None,
                             "filter_val": None, "filter_sigma": None},
                  "Y89": {"color": "#00FF00", "x_lower_bound": 0, "x_upper_bound": 1, "filter_type": None,
                          "filter_val": None, "filter_sigma": None}}
    channels_selected = ["Y89"]
    aliases = {}
    for key in blend_dict.keys():
        aliases[key] = key
    global_apply_filter = []
    global_filter_type = "gaussian"
    global_filter_val = 3
    global_filter_sigma = 1
    atlas_conn.insert_blend_config("test_config", blend_dict, channels_selected, global_apply_filter,
                                   global_filter_type, global_filter_val, global_filter_sigma, "roi_1")
    documents, config_names = atlas_conn.blend_configs_by_user()
    assert len(documents) == len(config_names) == 1
    assert config_names == ['test_config']
    atlas_conn.remove_blend_document_by_name('test_config')
    documents, config_names = atlas_conn.blend_configs_by_user()
    assert len(documents) == len(config_names) == 0
    assert not config_names
    atlas_conn.close()


@skip_on(ServerSelectionTimeoutError, "A connection to the mongoDB server could not be established")
def test_null_db_connection():
    fake_connection = AtlasDatabaseConnection(username="fake", password="fake")
    assert isinstance(fake_connection, AtlasDatabaseConnection)
    assert fake_connection.database is None
    assert fake_connection.blend_collection is None

    connect, message = fake_connection.create_connection()
    assert isinstance(fake_connection.database, pymongo.database.Database)
    assert isinstance(fake_connection.blend_collection, pymongo.collection.Collection)
    assert not connect

    with pytest.raises(OperationFailure):
        fake_connection.client.server_info()

    fake_connection.close()
    with pytest.raises(InvalidOperation):
        fake_connection.client.server_info()

    assert fake_connection.username_password_pair() == {'username': 'fake', 'password': 'fake'}
