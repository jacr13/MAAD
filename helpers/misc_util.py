import platform
from pathlib import Path


def is_docker():
    """Checks whether code is running inside docker container."""
    cgroup = Path("/proc/self/cgroup")
    return (
        Path("/.dockerenv").is_file()
        or cgroup.is_file()
        and cgroup.read_text().find("docker") > -1
    )


def get_running_platform():
    """Returns the running platform.

    Output:
        - platform name: ['linux', 'darwin', 'windows', 'docker']
    """

    platform_name = platform.system()
    if is_docker():
        return "docker"
    return platform_name.lower()


def zipsame(*seqs):
    """Verify that all the sequences in `seqs` are the same length, then zip them together"""
    assert seqs, "empty input sequence"
    ref_len = len(seqs[0])
    assert all(len(seq) == ref_len for seq in seqs[1:])
    return zip(*seqs)


def prettify_time(seconds, format_time="D-h:m:s"):
    """Print the number of seconds in human-readable format.
    Examples:
        '2 days', '2 hours and 37 minutes', 'less than a minute'.
    """
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24

    def helper(count, name):
        return "{} {}{}".format(str(count), name, ("s" if count != 1 else ""))

    if format_time == "default":
        # Display only the two greater units (days and hours, hours and minutes,
        # minutes and seconds)
        if days > 0:
            message = helper(days, "day")
            if hours > 0:
                message += " and " + helper(hours, "hour")
            return message
        if hours > 0:
            message = helper(hours, "hour")
            if minutes > 0:
                message += " and " + helper(minutes, "minute")
            return message
        if minutes > 0:
            return helper(minutes, "minute")
        # Finally, if none of the previous conditions is valid
        return "less than a minute"
    else:
        format_time = format_time.replace("D", str(int(days)))
        format_time = format_time.replace("h", f"{int(hours):02d}")
        format_time = format_time.replace("m", f"{int(minutes):02d}")
        format_time = format_time.replace("s", f"{int(seconds):02d}")
        return format_time


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Args:
        parser (argparse.Parser): Parser to add the flag to
        name (str): Name of the flag
          --<name> will enable the flag, while --no-<name> will disable it
        default (bool or None): Default value of the flag
        help (str): Help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name,
        action="store_true",
        default=default,
        dest=dest,
        help=help,
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)
