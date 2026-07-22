from db_ops.embeddings import update_embeddings


def update_to_collection(db, collection_name, data):
    try:
        result = db[collection_name].bulk_upsert(data)
        return True, result
    except Exception as e:
        print("Error writing local JSON collection:", e)
        return False, str(e)


def update_simulations(db, data):
    return update_to_collection(db, "simulations", data)


def update_simulation_sessions(db, data):
    return update_to_collection(db, "simulation_sessions", data)


def update_findings(db, data):
    return update_to_collection(db, "findings", data)
