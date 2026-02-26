newton.usd
==========

Utilities for working with the Universal Scene Description (USD) format.

This module provides both low-level USD utility helpers and public schema
resolver types used by :meth:`newton.ModelBuilder.add_usd`.

.. currentmodule:: newton.usd

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   PrimType
   SchemaResolver
   SchemaResolverMjc
   SchemaResolverNewton
   SchemaResolverPhysx

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   get_attribute
   get_attributes_in_namespace
   get_custom_attribute_declarations
   get_custom_attribute_values
   get_float
   get_gprim_axis
   get_mesh
   get_quat
   get_scale
   get_transform
   has_applied_api_schema
   has_attribute
   type_to_warp
   value_to_warp
