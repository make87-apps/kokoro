use std::error::Error;

use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use make87_messages::text::PlainText;

macro_rules! recv_and_print {
    ($sub:expr) => {{
        let subscriber = $sub;
        let message_encoder = ProtobufEncoder::<PlainText>::new();
        while let Ok(sample) = subscriber.recv_async().await {
            let message_decoded = message_encoder.decode(&sample.payload().to_bytes());
            match message_decoded {
                Ok(msg) => log::info!("Received: {:?}", msg),
                Err(e) => log::error!("Decode error: {e}"),
            }
        }
    }};
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let zenoh_interface = ZenohInterface::from_default_env("zenoh")?;
    let session = zenoh_interface.get_session().await?;

    let configured_subscriber = zenoh_interface
        .get_subscriber(&session, "incoming_message")
        .await?;

    match configured_subscriber {
        ConfiguredSubscriber::Fifo(sub) => recv_and_print!(&sub),
        ConfiguredSubscriber::Ring(sub) => recv_and_print!(&sub),
    }

    Ok(())
}
