# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os


def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))


def get_asset_directory() -> str:
    return os.path.join(get_source_directory(), "assets")


def get_asset(filename: str) -> str:
    return os.path.join(get_asset_directory(), filename)


def get_kamino_assets_directory() -> str:
    return os.path.join(get_asset_directory(), "kamino")


def get_kamino_basics_asset(filename: str) -> str:
    return os.path.join(get_kamino_assets_directory(), "basics", filename)


def get_kamino_testing_asset(filename: str) -> str:
    return os.path.join(get_kamino_assets_directory(), "testing", filename)
