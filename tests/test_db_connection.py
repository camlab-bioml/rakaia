import pytest
from ccramic.db.connection import AtlasDatabaseConnection
import pymongo
from pymongo.errors import OperationFailure, InvalidOperation

def test_db_connection():
    fake_connection = AtlasDatabaseConnection(username="fake", password="fake")
    assert isinstance(fake_connection, AtlasDatabaseConnection)
    assert isinstance(fake_connection.database, pymongo.database.Database)
    assert isinstance(fake_connection.blend_collection, pymongo.collection.Collection)

    with pytest.raises(OperationFailure):
        fake_connection.client.server_info()

    fake_connection.close()
    with pytest.raises(InvalidOperation):
        fake_connection.client.server_info()

    connect, message = fake_connection.ping_connection()
    assert not connect

    assert fake_connection.username_password_pair() == {'username': 'fake', 'password': 'fake'}
