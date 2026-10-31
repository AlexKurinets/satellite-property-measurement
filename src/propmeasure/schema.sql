CREATE SCHEMA IF NOT EXISTS pm;

CREATE TABLE IF NOT EXISTS pm.properties (
    property_id   text PRIMARY KEY,
    market        text NOT NULL,
    split         text NOT NULL CHECK (split IN ('base', 'train', 'val', 'test', 'live')),
    lat           double precision NOT NULL,
    lon           double precision NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- one capture per property, already on the Web Mercator grid and rotated so the frontage faces down
CREATE TABLE IF NOT EXISTS pm.captures (
    property_id   text PRIMARY KEY REFERENCES pm.properties ON DELETE CASCADE,
    captured_on   date NOT NULL,
    season        text NOT NULL,
    zoom          int NOT NULL,
    origin_x      double precision NOT NULL,
    origin_y      double precision NOT NULL,
    png           bytea NOT NULL
);

-- county parcel polygon as delivered (lon/lat), offset from the imagery by whatever the county's GIS is off by
CREATE TABLE IF NOT EXISTS pm.parcels (
    property_id   text PRIMARY KEY REFERENCES pm.properties ON DELETE CASCADE,
    ring          jsonb NOT NULL,
    source        text NOT NULL DEFAULT 'county'
);

CREATE TABLE IF NOT EXISTS pm.pitch_observations (
    property_id   text PRIMARY KEY REFERENCES pm.properties ON DELETE CASCADE,
    rise_in_12    real NOT NULL,
    source        text NOT NULL DEFAULT 'lidar'
);

-- traced class masks; label_set says which class list the indices follow
CREATE TABLE IF NOT EXISTS pm.labels (
    property_id   text PRIMARY KEY REFERENCES pm.properties ON DELETE CASCADE,
    label_set     text NOT NULL CHECK (label_set IN ('base', 'client')),
    mask_png      bytea NOT NULL
);

-- measured truth: 'traced' for the benchmark and validation sets; 'synthetic' drives the simulated estimators
CREATE TABLE IF NOT EXISTS pm.ground_truth (
    property_id        text PRIMARY KEY REFERENCES pm.properties ON DELETE CASCADE,
    lawn_sqft          double precision NOT NULL,
    bed_sqft           double precision NOT NULL,
    pavement_sqft      double precision NOT NULL,
    roof_sqft          double precision NOT NULL,
    roof_plan_sqft     double precision NOT NULL,
    roof_perimeter_ft  double precision NOT NULL,
    method             text NOT NULL CHECK (method IN ('traced', 'synthetic'))
);

CREATE TABLE IF NOT EXISTS pm.models (
    version         text PRIMARY KEY,
    parent_version  text REFERENCES pm.models,
    label_set       text NOT NULL,
    pipeline        text NOT NULL,
    weights         bytea NOT NULL,
    train_config    jsonb NOT NULL,
    history         jsonb NOT NULL DEFAULT '[]',
    status          text NOT NULL CHECK (status IN ('candidate', 'production', 'retired')),
    created_at      timestamptz NOT NULL DEFAULT now(),
    promoted_at     timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS models_one_production ON pm.models (status) WHERE status = 'production';

CREATE TABLE IF NOT EXISTS pm.eval_runs (
    id              bigserial PRIMARY KEY,
    model_version   text REFERENCES pm.models,  -- null for an oracle run on traced masks
    pipeline        text NOT NULL,
    benchmark       text NOT NULL,
    report          jsonb NOT NULL,
    gate            jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pm.measurements (
    id              bigserial PRIMARY KEY,
    property_id     text NOT NULL REFERENCES pm.properties ON DELETE CASCADE,
    model_version   text NOT NULL REFERENCES pm.models,
    pipeline        text NOT NULL,
    result          jsonb NOT NULL,
    review_reasons  jsonb NOT NULL DEFAULT '[]',
    measured_on     date NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS measurements_property ON pm.measurements (property_id, created_at DESC);

-- what an estimator changed in the quoting screen before sending a quote
CREATE TABLE IF NOT EXISTS pm.corrections (
    id               bigserial PRIMARY KEY,
    measurement_id   bigint NOT NULL REFERENCES pm.measurements ON DELETE CASCADE,
    feature          text NOT NULL CHECK (feature IN ('lawn_sqft', 'bed_sqft', 'pavement_sqft', 'roof_sqft',
                                                      'roof_perimeter_ft')),
    measured_value   double precision NOT NULL,
    corrected_value  double precision NOT NULL CHECK (corrected_value >= 0),
    corrected_by     text NOT NULL,
    note             text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (measurement_id, feature)
);
