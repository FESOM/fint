import os


def update_attrs(ds, args):
    if args.target is not None:
        ds["target"] = args.target

    ds["box"] = args.box
    ds["influence"] = args.influence
    ds["interp"] = args.interp
    ds["data"] = os.path.abspath(args.data)
    ds["meshpath"] = os.path.abspath(args.meshpath)
    return ds
