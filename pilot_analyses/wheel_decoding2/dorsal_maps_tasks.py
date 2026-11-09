"""Task collection lookup (as dorsal_maps/extract_meso.task_collection) without importing extract_meso."""


def task_collection_for(one, eid):
    cols = sorted({d.rsplit("/", 1)[0] for d in one.list_datasets(eid) if "_ibl_trials." in d})
    if not cols:
        raise FileNotFoundError("no trials object")
    return cols[0]
