import atexit
import logging
import os
import tempfile
import struct


logger = logging.getLogger(__name__)


def discover_bins(path, srcbuild=False):
    if not os.path.exists(path):
        raise FileNotFoundError("path does not exist")

    if os.path.isfile(path):
        if os.path.basename(path) == "boot.bin":
            return {"boot": path}
        raise ValueError("file is not a valid binary")

    retrieved_bins = {}
    bin_dict = {
        "boot": ["boot"],
        "gateware": ["gateware"],
        "bootloader": ["bootloader"],
        "firmware": ["runtime", "satman"],
    }

    for name, components in bin_dict.items():
        try:
            retrieved_bins[name] = fetch_bin(path, components, srcbuild)
        except FileNotFoundError:
            pass
    return retrieved_bins


def artifact_path(this_binary_dir, *path_filename, srcbuild=False):
    if srcbuild:
        # source tree - use path elements to locate file
        return os.path.join(this_binary_dir, *path_filename)
    else:
        # flat tree - all files in the same directory, discard path elements
        *_, filename = path_filename
        return os.path.join(this_binary_dir, filename)


def fetch_bin(binary_dir, components, srcbuild=False):
    if len(components) > 1:
        bins = []
        for option in components:
            try:
                bins.append(fetch_bin(binary_dir, [option], srcbuild))
            except FileNotFoundError:
                pass

        if len(bins) == 0:
            raise FileNotFoundError("multiple components not found: {}".format(
                                        " ".join(components)))
        
        if len(bins) > 1:
            raise ValueError("more than one file, "
                             "please clean up your build directory. "
                             "Found files: {}".format(
                             " ".join(bins)))

        return bins[0]

    else:
        component = components[0]
        path = artifact_path(binary_dir, *{
            "gateware": ["gateware", "top.bin"],
            "boot": ["boot.bin"],
            "bootloader": ["software", "bootloader", "bootloader.bin"],
            "runtime": ["software", "runtime", "runtime.fbi"],
            "satman": ["software", "satman", "satman.fbi"],
        }[component], srcbuild=srcbuild)

        if not os.path.exists(path):
            raise FileNotFoundError("{} not found".format(component))

        return path
