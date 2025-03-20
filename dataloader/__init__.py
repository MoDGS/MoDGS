from dataloader.timePcdTable_dataset import TimePCDTable,NeighbourFlowPairsDataset,ExhaustiveFlowPairsDataset
TIMETABLE_DATASET={
    "TimePCDTable":TimePCDTable,
    "NeighbourFlowPairsDataset":NeighbourFlowPairsDataset,
    "ExhaustiveFlowPairsDataset":ExhaustiveFlowPairsDataset
}
def get_dataset(dataset_name:str):
    return TIMETABLE_DATASET[dataset_name]
def get_dataset_by_table_dir(dir:str,exhaustive_data=False):
    if dir.endswith(".npy") or dir.endswith(".npz"):
        print("ends with .npy or .npz, assuming -TimePCDTable- dataset!!")
        return TimePCDTable
    elif dir.endswith(".pkl") and  not exhaustive_data :
        print("ends with .pkl, assuming -NeighbourFlowPairsDataset- dataset!!")
        return NeighbourFlowPairsDataset
    elif dir.endswith(".pkl") and exhaustive_data==True:
        print("ends with .pkl, exhaustive_data==True  assuming -ExhaustiveFlowPairsDataset- dataset!!")
        return ExhaustiveFlowPairsDataset
    else:
        raise ValueError("Unknown dataset type")