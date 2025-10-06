from django.db import models

class Station(models.Model):
    id = models.CharField(primary_key=True, max_length=200)
    name = models.TextField()
    tz = models.TextField()
    region = models.TextField()
    stream_url = models.TextField()

    class Meta:
        managed = False
        db_table = "stations"

class Chunk(models.Model):
    id = models.CharField(primary_key=True, max_length=200)
    station = models.ForeignKey(Station, to_field="id",
                                db_column="station_id",
                                on_delete=models.DO_NOTHING)
    t0 = models.DateTimeField()
    t1 = models.DateTimeField()
    speaker = models.TextField(null=True)
    text = models.TextField(null=True)
    clip_uri = models.TextField(null=True)

    class Meta:
        managed = False
        db_table = "chunks"
        indexes = [models.Index(fields=["t0"])]

class Entity(models.Model):
    id = models.BigAutoField(primary_key=True)
    chunk = models.ForeignKey(Chunk, to_field="id",
                              db_column="chunk_id",
                              on_delete=models.DO_NOTHING,
                              related_name="entities")
    typ = models.TextField(db_column="typ")
    surface = models.TextField(db_column="surface")
    qid = models.TextField(null=True)
    conf = models.FloatField(null=True)
    lat = models.FloatField(null=True)
    lon = models.FloatField(null=True)

    class Meta:
        managed = False
        db_table = "entities"
