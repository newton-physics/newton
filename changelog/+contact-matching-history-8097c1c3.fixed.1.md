Fix `Contacts.rigid_contact_match_index` keeping a previous pipeline's match indices when a `CollisionPipeline` without contact matching writes the same buffer; every index now reads as unmatched.
