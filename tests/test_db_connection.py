from ccramic.db.connection import AtlasDatabaseConnection
import pymongo

def test_db_connection():
    fake_connection = AtlasDatabaseConnection(username="fake", password="fake")
    assert isinstance(fake_connection, AtlasDatabaseConnection)
    assert isinstance(fake_connection.database, pymongo.database.Database)
    assert isinstance(fake_connection.blend_collection, pymongo.collection.Collection)

    connect, message = fake_connection.ping_connection()
    assert not connect

    assert fake_connection.username_password_pair() == {'username': 'fake', 'password': 'fake'}
