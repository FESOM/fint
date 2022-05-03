import os

def update_attrs(ds, args):
        if args.target is not None:
            ds["target"] = args.target
        
        ds["box"] = args.box
        # ds["res"] = args.res
        # ds.attrs["map_projection"] = args.map_projection
        ds["influence"] = args.influence
        ds["interp"] = args.interp
        # ds.attrs["depths"] = args.depths
        # ds.attrs["timesteps"] = args.timesteps
        ds["data"] = os.path.abspath(args.data)
        ds["meshpath"] = os.path.abspath(args.meshpath)
        # ds.attrs["ofile"] = out_file
        # ds.attrs["odir"] = args.odir
        return ds