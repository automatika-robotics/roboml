from typing import Optional
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings

from roboml.interfaces import DBAdd, DBMetadataQuery, DBQuery
from roboml.models._encoding import EncodingModel
from roboml.ray import app, ingress_decorator

from ._base import VectorDBTemplate


@ingress_decorator
class ChromaDB(VectorDBTemplate):
    """
    ChromaDB Wrapper.
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.vectordb: ClientAPI

    @app.post("/initialize")
    def _initialize(
        self,
        db_location: str = "./data",
        username: Optional[str] = None,
        password: Optional[str] = None,
        encoder: Optional[dict] = None,
    ) -> None:
        """
        Initializes the db.
        """
        if username and password:
            self.logger.warning(
                "Username/password authentication can only be used with local ChromaDB client. Cannot use username/password authentication with roboml ChromaDB instance."
            )
        # initialize the encoding model
        self.encoding_model = EncodingModel(name="encoding_model")
        encoder = encoder or {}
        self.encoding_model._initialize(**encoder)

        # create a vectordb client
        self.vectordb = PersistentClient(
            settings=Settings(anonymized_telemetry=False),
            path=db_location,
        )

    @app.post("/add")
    def _add(self, data: DBAdd) -> dict:
        """Add data to the given collection.
        :param data:
        :param type: DBAdd
        :rtype: dict
        """
        # If specified reset existing collection
        if data.reset_collection:
            try:
                self.vectordb.delete_collection(name=data.collection_name)
            except ValueError:
                self.logger.warning(
                    f"Cannot delete collection with name {data.collection_name} as it does not exist."
                )
        try:
            # create a collection if one doesnt exist
            collection = self.vectordb.get_or_create_collection(
                name=data.collection_name, metadata={"hnsw:space": data.distance_func}
            )
            # create embeddings
            embeddings = self.encoding_model.embed_documents(data.documents)
            # add to collection
            collection.add(
                documents=data.documents,
                embeddings=embeddings,
                metadatas=data.metadatas,
                ids=data.ids,
            )
        except Exception as e:
            self.logger.error(f"Exception occured: {e}")
            raise

        return {"output": "Success"}

    @app.post("/conditional_add")
    def _conditional_add(self, data: DBAdd) -> dict:
        """First check if id exists if not then add data to the collection provided
        Update metadatas of the ids that exist
        :param data:
        :param type: DBAdd
        :rtype: dict
        """
        try:
            # create a collection if one doesnt exist
            collection = self.vectordb.get_or_create_collection(
                name=data.collection_name, metadata={"hnsw:space": data.distance_func}
            )
            # check for ids that already exist in DB
            already_existing = collection.get(ids=data.ids)

            # if ids found in database, remove them from main input lists
            # update metadatas of already existing IDs
            metadatas_to_update = []
            to_be_deleted = []
            if already_existing_ids := already_existing["ids"]:
                for idx in range(len(data.ids)):
                    if data.ids[idx] in already_existing_ids:
                        to_be_deleted.append(idx)
                        metadatas_to_update.append(data.metadatas[idx].copy())
                # do the metadata update
                collection.update(
                    ids=already_existing_ids, metadatas=metadatas_to_update
                )
        except Exception as e:
            self.logger.error(f"Exception occured: {e}")
            raise

        # delete from indices that were updated
        for idx in sorted(to_be_deleted, reverse=True):
            del data.ids[idx]
            del data.metadatas[idx]
            del data.documents[idx]
        # add the remaining data
        return self._add(data) if data.ids else {"output": "Success"}

    @app.post("/metadata_query")
    def _metadata_query(self, data: DBMetadataQuery) -> dict:
        """Retreive data by metadata query.
        :param data:
        :param type: DBMetadataQuery
        :rtype: dict
        """
        # create filters for all metadata values
        all_filters = []
        for metadata in data.metadatas:
            # create filter for each metadata hashmap
            if len(metadata) > 1:
                filter = {"$and": [{i: {"$eq": metadata[i]}} for i in metadata]}
            # if there is only one metadata entry, $and is not needed
            elif len(metadata) == 1:
                filter = {list(metadata.keys())[0]: {"$eq": list(metadata.values())[0]}}
            else:
                continue
            all_filters.append(filter)

        # if no filters, return output as None
        if len(all_filters) == 0:
            self.logger.warning(
                "The metadata filters received were empty, please call query method for retreiving data without metadata filtering."
            )
            return {"output": []}

        # if there are multiple filters, add an $or
        filters = {"$or": all_filters} if len(all_filters) > 1 else all_filters[0]

        try:
            # get collection
            collection = self.vectordb.get_collection(name=data.collection_name)
            # get filtered data
            output = collection.get(where=filters)
        except Exception as e:
            self.logger.error(f"Exception occured: {e}")
            raise
        return {"output": output}

    @app.post("/query")
    def _query(self, data: DBQuery) -> dict:
        """
        Retreives results for a given DB query
        :param data:
        :param type: DBQuery
        :rtype: dict
        """
        query_vec = self.encoding_model.embed_query(data.query)
        query_vec = query_vec.tolist()
        try:
            # create a collection for the map data
            collection = self.vectordb.get_collection(name=data.collection_name)
            output = (
                collection.query(query_embeddings=query_vec, n_results=data.n_results)
                or []
            )
        except Exception as e:
            self.logger.error(f"Exception occured: {e}")
            raise

        return {"output": output}
