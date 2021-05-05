splicemachine.features package
===========================

Submodules
----------
..
  .. automodule:: splicemachine.features.feature_store
     :members:
     :undoc-members:
     :show-inheritance:

  .. automodule:: splicemachine.features.feature_set
     :members:
     :undoc-members:
     :show-inheritance:

  .. automodule:: splicemachine.features.feature
     :members:
     :undoc-members:
     :show-inheritance:

  .. automodule:: splicemachine.features.training_view
     :members:
     :undoc-members:
     :show-inheritance:

splicemachine.features.feature_store module
-------------------------------------------

This Module contains the classes and APIs for interacting with the Splice Machine Feature Store.

.. automodule:: splicemachine.features.feature_store
   :members:
   :undoc-members:
   :show-inheritance:

splicemachine.features.feature_set
----------------------------------

This describes the Python representation of a Feature Set. A feature set is a database table that contains Features and their metadata.
The Feature Set class is mostly used internally but can be used by the user to see the available Features in the given
Feature Set, to see the table and schema name it is deployed to (if it is deployed), and to deploy the feature set
(which can also be done directly through the Feature Store). Feature Sets are unique by their schema.table name, as they
exist in the Splice Machine database as a SQL table. They are case insensitive.
To see the full contents of your Feature Set, you can print, return, or .__dict__ your Feature Set object.

.. automodule:: splicemachine.features.feature_set
   :members:
   :show-inheritance:


splicemachine.features.Feature
----------------------------------

This describes the Python representation of a Feature. A Feature is a column of a Feature Set table with particular metadata.
A Feature is the smallest unit in the Feature Store, and each Feature within a Feature Set is individually tracked for changes
to enable full time travel and point-in-time consistent training datasets. Features' names are unique and case insensitive.
To see the full contents of your Feature, you can print, return, or .__dict__ your Feature object.

.. automodule:: splicemachine.features.feature
   :members:
   :undoc-members:
   :show-inheritance:

splicemachine.features.training_view
----------------------------------

This describes the Python representation of a Training View. A Training View is a SQL statement defining an event of interest, and metadata around how to create a training dataset with that view.
To see the full contents of your Training View, you can print, return, or .__dict__ your Training View object.

.. automodule:: splicemachine.features.training_view
   :members:
   :undoc-members:
   :show-inheritance:


Module contents
---------------

.. automodule:: splicemachine.features
   :members:
   :undoc-members:
   :show-inheritance:
