//! Exact asynchronous order-lifecycle state machine for execution runtimes.
//!
//! The runtime owns `local_sequence`, so local journal ordering is contiguous.
//! `native_sequence` is retained as venue evidence only: venue adapters own any
//! protocol-specific gap/reset semantics. Economic effects use the exact base-10
//! financial types and stable fill identities rather than floating tolerances.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::financial::{ExactOrderRequest, Price, Quantity, SignedAmount};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FillKey {
    pub venue: String,
    pub account_id: String,
    pub instrument_id: String,
    pub trade_id: String,
}

impl FillKey {
    pub fn validate(&self) -> bool {
        !self.venue.is_empty()
            && !self.account_id.is_empty()
            && !self.instrument_id.is_empty()
            && !self.trade_id.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FillRecordV2 {
    pub key: FillKey,
    pub client_order_id: String,
    pub occurred_at_ms: i64,
    pub price: Price,
    pub quantity: Quantity,
    pub fee_asset: String,
    pub fee_amount: SignedAmount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatusV2 {
    PendingSubmit,
    SubmitUnknown,
    Open,
    PartiallyFilled,
    PendingCancel,
    CancelUnknown,
    PendingAmend,
    AmendUnknown,
    Filled,
    Canceled,
    Rejected,
    ReconciliationRequired,
}

impl ExecutionStatusV2 {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Filled | Self::Canceled | Self::Rejected)
    }

    pub fn is_ambiguous(self) -> bool {
        matches!(
            self,
            Self::SubmitUnknown
                | Self::CancelUnknown
                | Self::AmendUnknown
                | Self::ReconciliationRequired
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionOrderV2 {
    pub intent_id: String,
    pub idempotency_key: String,
    pub agent_id: String,
    pub account_id: String,
    pub venue: String,
    pub client_order_id: String,
    pub request: ExactOrderRequest,
    pub exchange_order_id: Option<String>,
    pub status: ExecutionStatusV2,
    /// Exact accumulated fill quantity. Kept as a zero-capable amount because a
    /// positive-only `Quantity` cannot represent an unfilled order.
    pub filled_quantity: SignedAmount,
    /// Exact signed fees/rebates keyed by their native asset.
    pub fees_by_asset: BTreeMap<String, SignedAmount>,
    pub revision: u64,
    pub last_event_ts_ms: i64,
    pub cancel_effective_ts_ms: Option<i64>,
    pub unresolved_reason: Option<String>,
}

impl ExecutionOrderV2 {
    pub fn remaining_quantity(&self) -> Result<SignedAmount, LifecycleV2Error> {
        let requested = SignedAmount::from(self.request.quantity);
        let remaining = requested
            .checked_sub(self.filled_quantity)
            .map_err(|_| LifecycleV2Error::ArithmeticOverflow)?;
        if remaining.is_negative()
        {
            return Err(LifecycleV2Error::Overfill);
        }
        Ok(remaining)
    }

    /// A timeout/unknown submit result is never evidence that repeating the
    /// external submission is safe. Reconciliation by stable identity comes first.
    pub const fn may_repeat_submit(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionEventKindV2 {
    Accepted {
        exchange_order_id: String,
    },
    SubmitRejected {
        reason: String,
    },
    /// External result is ambiguous. Query/reconcile by client/idempotency
    /// identity before any new submission.
    SubmitUnknown {
        reason: String,
    },
    Fill {
        key: FillKey,
        occurred_at_ms: i64,
        price: Price,
        quantity: Quantity,
        fee_asset: String,
        fee_amount: SignedAmount,
    },
    CancelRequested,
    Canceled {
        effective_at_ms: i64,
    },
    CancelRejected {
        reason: String,
    },
    CancelUnknown {
        reason: String,
    },
    AmendRequested,
    Amended {
        request: ExactOrderRequest,
    },
    AmendRejected {
        reason: String,
    },
    AmendUnknown {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionEventV2 {
    /// Runtime-owned durable journal sequence. Must be exactly contiguous.
    pub local_sequence: u64,
    /// Venue/native sequence when available. No generic continuity assumption.
    pub native_sequence: Option<u64>,
    pub received_at_ms: i64,
    pub client_order_id: String,
    pub kind: ExecutionEventKindV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOutcomeV2 {
    Applied,
    DuplicateFill,
    ReconciliationRequired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleV2Error {
    InvalidIdentity,
    DuplicateIntentId(String),
    DuplicateIdempotencyKey(String),
    DuplicateClientOrderId(String),
    UnknownClientOrderId(String),
    LocalSequenceGap {
        expected: u64,
        incoming: u64,
    },
    InvalidTransition {
        client_order_id: String,
        status: ExecutionStatusV2,
    },
    InvalidFill,
    FillIdentityConflict(FillKey),
    Overfill,
    InvalidAmend,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleBookV2 {
    pub venue: String,
    pub account_id: String,
    pub last_local_sequence: u64,
    pub last_native_sequence: Option<u64>,
    pub orders: BTreeMap<String, ExecutionOrderV2>,
    pub fills: BTreeMap<FillKey, FillRecordV2>,
    intent_ids: BTreeSet<String>,
    idempotency_keys: BTreeSet<String>,
}

impl LifecycleBookV2 {
    pub fn new(venue: &str, account_id: &str) -> Result<Self, LifecycleV2Error> {
        if venue.is_empty() || account_id.is_empty()
        {
            return Err(LifecycleV2Error::InvalidIdentity);
        }
        Ok(Self {
            venue: venue.to_string(),
            account_id: account_id.to_string(),
            last_local_sequence: 0,
            last_native_sequence: None,
            orders: BTreeMap::new(),
            fills: BTreeMap::new(),
            intent_ids: BTreeSet::new(),
            idempotency_keys: BTreeSet::new(),
        })
    }

    /// Register an intent before external send. The owning durable runtime must
    /// persist this state in its transaction before performing network I/O.
    pub fn register_intent(
        &mut self,
        intent_id: String,
        idempotency_key: String,
        agent_id: String,
        client_order_id: String,
        request: ExactOrderRequest,
        ts_ms: i64,
    ) -> Result<(), LifecycleV2Error> {
        if intent_id.is_empty()
            || idempotency_key.is_empty()
            || agent_id.is_empty()
            || client_order_id.is_empty()
            || request.instrument_id.is_empty()
            || request.rules_version.is_empty()
        {
            return Err(LifecycleV2Error::InvalidIdentity);
        }
        if self.intent_ids.contains(&intent_id)
        {
            return Err(LifecycleV2Error::DuplicateIntentId(intent_id));
        }
        if self.idempotency_keys.contains(&idempotency_key)
        {
            return Err(LifecycleV2Error::DuplicateIdempotencyKey(idempotency_key));
        }
        if self.orders.contains_key(&client_order_id)
        {
            return Err(LifecycleV2Error::DuplicateClientOrderId(client_order_id));
        }

        self.intent_ids.insert(intent_id.clone());
        self.idempotency_keys.insert(idempotency_key.clone());
        self.orders.insert(
            client_order_id.clone(),
            ExecutionOrderV2 {
                intent_id,
                idempotency_key,
                agent_id,
                account_id: self.account_id.clone(),
                venue: self.venue.clone(),
                client_order_id,
                request,
                exchange_order_id: None,
                status: ExecutionStatusV2::PendingSubmit,
                filled_quantity: SignedAmount::ZERO,
                fees_by_asset: BTreeMap::new(),
                revision: 0,
                last_event_ts_ms: ts_ms,
                cancel_effective_ts_ms: None,
                unresolved_reason: None,
            },
        );
        Ok(())
    }

    pub fn apply_event(
        &mut self,
        event: ExecutionEventV2,
    ) -> Result<ApplyOutcomeV2, LifecycleV2Error> {
        let expected = self
            .last_local_sequence
            .checked_add(1)
            .ok_or(LifecycleV2Error::ArithmeticOverflow)?;
        if event.local_sequence != expected
        {
            return Err(LifecycleV2Error::LocalSequenceGap {
                expected,
                incoming: event.local_sequence,
            });
        }
        if event.client_order_id.is_empty()
        {
            return Err(LifecycleV2Error::InvalidIdentity);
        }

        let local_sequence = event.local_sequence;
        let native_sequence = event.native_sequence;
        let received_at_ms = event.received_at_ms;
        let client_order_id = event.client_order_id;

        let outcome = match event.kind
        {
            ExecutionEventKindV2::Accepted { exchange_order_id } =>
            {
                self.apply_accepted(&client_order_id, exchange_order_id, received_at_ms)?
            },
            ExecutionEventKindV2::SubmitRejected { reason } =>
            {
                self.apply_submit_rejected(&client_order_id, reason, received_at_ms)?
            },
            ExecutionEventKindV2::SubmitUnknown { reason } =>
            {
                let order = self.order_mut(&client_order_id)?;
                if order.status != ExecutionStatusV2::PendingSubmit
                {
                    return Err(invalid_transition(order));
                }
                order.status = ExecutionStatusV2::SubmitUnknown;
                order.unresolved_reason = Some(reason);
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::ReconciliationRequired
            },
            ExecutionEventKindV2::Fill {
                key,
                occurred_at_ms,
                price,
                quantity,
                fee_asset,
                fee_amount,
            } => self.apply_fill(
                &client_order_id,
                received_at_ms,
                key,
                occurred_at_ms,
                price,
                quantity,
                fee_asset,
                fee_amount,
            )?,
            ExecutionEventKindV2::CancelRequested =>
            {
                let order = self.order_mut(&client_order_id)?;
                if !matches!(
                    order.status,
                    ExecutionStatusV2::Open | ExecutionStatusV2::PartiallyFilled
                )
                {
                    return Err(invalid_transition(order));
                }
                order.status = ExecutionStatusV2::PendingCancel;
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::Applied
            },
            ExecutionEventKindV2::Canceled { effective_at_ms } =>
            {
                let post_cancel_fill_exists = self.fills.values().any(|fill| {
                    fill.client_order_id == client_order_id && fill.occurred_at_ms > effective_at_ms
                });
                let order = self.order_mut(&client_order_id)?;
                if !matches!(
                    order.status,
                    ExecutionStatusV2::Open
                        | ExecutionStatusV2::PartiallyFilled
                        | ExecutionStatusV2::PendingCancel
                        | ExecutionStatusV2::CancelUnknown
                )
                {
                    return Err(invalid_transition(order));
                }
                order.cancel_effective_ts_ms = Some(effective_at_ms);
                order.last_event_ts_ms = received_at_ms;
                if post_cancel_fill_exists
                {
                    order.status = ExecutionStatusV2::ReconciliationRequired;
                    order.unresolved_reason = Some(
                        "recorded fill occurred after confirmed cancellation timestamp".to_string(),
                    );
                    ApplyOutcomeV2::ReconciliationRequired
                }
                else
                {
                    order.status = ExecutionStatusV2::Canceled;
                    order.unresolved_reason = None;
                    ApplyOutcomeV2::Applied
                }
            },
            ExecutionEventKindV2::CancelRejected { reason } =>
            {
                let order = self.order_mut(&client_order_id)?;
                if !matches!(
                    order.status,
                    ExecutionStatusV2::PendingCancel | ExecutionStatusV2::CancelUnknown
                )
                {
                    return Err(invalid_transition(order));
                }
                order.status = economic_open_status(order)?;
                order.unresolved_reason = Some(reason);
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::Applied
            },
            ExecutionEventKindV2::CancelUnknown { reason } =>
            {
                let order = self.order_mut(&client_order_id)?;
                if order.status != ExecutionStatusV2::PendingCancel
                {
                    return Err(invalid_transition(order));
                }
                order.status = ExecutionStatusV2::CancelUnknown;
                order.unresolved_reason = Some(reason);
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::ReconciliationRequired
            },
            ExecutionEventKindV2::AmendRequested =>
            {
                let order = self.order_mut(&client_order_id)?;
                if !matches!(
                    order.status,
                    ExecutionStatusV2::Open | ExecutionStatusV2::PartiallyFilled
                )
                {
                    return Err(invalid_transition(order));
                }
                order.status = ExecutionStatusV2::PendingAmend;
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::Applied
            },
            ExecutionEventKindV2::Amended { request } =>
            {
                self.apply_amended(&client_order_id, request, received_at_ms)?
            },
            ExecutionEventKindV2::AmendRejected { reason } =>
            {
                let order = self.order_mut(&client_order_id)?;
                if !matches!(
                    order.status,
                    ExecutionStatusV2::PendingAmend | ExecutionStatusV2::AmendUnknown
                )
                {
                    return Err(invalid_transition(order));
                }
                order.status = economic_open_status(order)?;
                order.unresolved_reason = Some(reason);
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::Applied
            },
            ExecutionEventKindV2::AmendUnknown { reason } =>
            {
                let order = self.order_mut(&client_order_id)?;
                if order.status != ExecutionStatusV2::PendingAmend
                {
                    return Err(invalid_transition(order));
                }
                order.status = ExecutionStatusV2::AmendUnknown;
                order.unresolved_reason = Some(reason);
                order.last_event_ts_ms = received_at_ms;
                ApplyOutcomeV2::ReconciliationRequired
            },
        };

        self.last_local_sequence = local_sequence;
        self.last_native_sequence = native_sequence.or(self.last_native_sequence);
        Ok(outcome)
    }

    fn apply_accepted(
        &mut self,
        client_order_id: &str,
        exchange_order_id: String,
        received_at_ms: i64,
    ) -> Result<ApplyOutcomeV2, LifecycleV2Error> {
        if exchange_order_id.is_empty()
        {
            return Err(LifecycleV2Error::InvalidIdentity);
        }
        let order = self.order_mut(client_order_id)?;
        if let Some(existing) = &order.exchange_order_id
        {
            if existing != &exchange_order_id
            {
                order.status = ExecutionStatusV2::ReconciliationRequired;
                order.unresolved_reason =
                    Some("conflicting exchange order identifiers".to_string());
                order.last_event_ts_ms = received_at_ms;
                return Ok(ApplyOutcomeV2::ReconciliationRequired);
            }
        }
        else
        {
            order.exchange_order_id = Some(exchange_order_id);
        }

        if order.status == ExecutionStatusV2::Rejected
        {
            order.status = ExecutionStatusV2::ReconciliationRequired;
            order.unresolved_reason = Some("accepted after submit rejection".to_string());
            order.last_event_ts_ms = received_at_ms;
            return Ok(ApplyOutcomeV2::ReconciliationRequired);
        }
        if matches!(
            order.status,
            ExecutionStatusV2::PendingSubmit | ExecutionStatusV2::SubmitUnknown
        )
        {
            order.status = economic_open_status(order)?;
            order.unresolved_reason = None;
        }
        order.last_event_ts_ms = received_at_ms;
        Ok(ApplyOutcomeV2::Applied)
    }

    fn apply_submit_rejected(
        &mut self,
        client_order_id: &str,
        reason: String,
        received_at_ms: i64,
    ) -> Result<ApplyOutcomeV2, LifecycleV2Error> {
        let order = self.order_mut(client_order_id)?;
        if !order.filled_quantity.is_zero() || order.exchange_order_id.is_some()
        {
            order.status = ExecutionStatusV2::ReconciliationRequired;
            order.unresolved_reason =
                Some(format!("submit rejected after venue evidence: {reason}"));
            order.last_event_ts_ms = received_at_ms;
            return Ok(ApplyOutcomeV2::ReconciliationRequired);
        }
        if !matches!(
            order.status,
            ExecutionStatusV2::PendingSubmit | ExecutionStatusV2::SubmitUnknown
        )
        {
            return Err(invalid_transition(order));
        }
        order.status = ExecutionStatusV2::Rejected;
        order.unresolved_reason = Some(reason);
        order.last_event_ts_ms = received_at_ms;
        Ok(ApplyOutcomeV2::Applied)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_fill(
        &mut self,
        client_order_id: &str,
        received_at_ms: i64,
        key: FillKey,
        occurred_at_ms: i64,
        price: Price,
        quantity: Quantity,
        fee_asset: String,
        fee_amount: SignedAmount,
    ) -> Result<ApplyOutcomeV2, LifecycleV2Error> {
        if !key.validate()
            || key.venue != self.venue
            || key.account_id != self.account_id
            || fee_asset.is_empty()
        {
            return Err(LifecycleV2Error::InvalidFill);
        }

        let incoming = FillRecordV2 {
            key: key.clone(),
            client_order_id: client_order_id.to_string(),
            occurred_at_ms,
            price,
            quantity,
            fee_asset: fee_asset.clone(),
            fee_amount,
        };
        if let Some(existing) = self.fills.get(&key)
        {
            if existing == &incoming
            {
                return Ok(ApplyOutcomeV2::DuplicateFill);
            }
            return Err(LifecycleV2Error::FillIdentityConflict(key));
        }

        let order = self.order_mut(client_order_id)?;
        if key.instrument_id != order.request.instrument_id
            || order.status == ExecutionStatusV2::Rejected
        {
            return Err(LifecycleV2Error::InvalidFill);
        }

        let status_before = order.status;
        let requested = SignedAmount::from(order.request.quantity);
        let new_filled_quantity = order
            .filled_quantity
            .checked_add_quantity(quantity)
            .map_err(|_| LifecycleV2Error::ArithmeticOverflow)?;
        let new_remaining = requested
            .checked_sub(new_filled_quantity)
            .map_err(|_| LifecycleV2Error::ArithmeticOverflow)?;
        if new_remaining.is_negative()
        {
            return Err(LifecycleV2Error::Overfill);
        }
        let current_fee = order
            .fees_by_asset
            .get(&fee_asset)
            .copied()
            .unwrap_or(SignedAmount::ZERO);
        let total_fee = current_fee
            .checked_add(fee_amount)
            .map_err(|_| LifecycleV2Error::ArithmeticOverflow)?;
        let contradictory_after_cancel = order
            .cancel_effective_ts_ms
            .is_some_and(|canceled_at| occurred_at_ms > canceled_at);

        // No economic state is mutated until all fallible arithmetic above has
        // completed. An error therefore leaves the event safe to retry.
        order.filled_quantity = new_filled_quantity;
        order.fees_by_asset.insert(fee_asset, total_fee);
        order.last_event_ts_ms = received_at_ms;

        if contradictory_after_cancel
        {
            order.status = ExecutionStatusV2::ReconciliationRequired;
            order.unresolved_reason =
                Some("fill occurred after confirmed cancellation timestamp".to_string());
        }
        else if status_before == ExecutionStatusV2::ReconciliationRequired
        {
            // Contradictory venue evidence remains unresolved until explicit
            // reconciliation; later economic events must not silently clear it.
        }
        else if matches!(
            status_before,
            ExecutionStatusV2::PendingAmend | ExecutionStatusV2::AmendUnknown
        )
        {
            // Even a fill of the current quantity cannot resolve an outstanding
            // amendment: a later confirmed increase can make the order open again.
        }
        else if new_remaining.is_zero()
        {
            order.status = ExecutionStatusV2::Filled;
            order.unresolved_reason = None;
        }
        else if status_before == ExecutionStatusV2::Canceled
        {
            // The fill occurred before cancellation but was delivered later.
            // Preserve Canceled for the remaining quantity.
        }
        else if matches!(
            status_before,
            ExecutionStatusV2::PendingCancel | ExecutionStatusV2::CancelUnknown
        )
        {
            // Preserve pending/unknown cancellation while accounting the fill.
        }
        else
        {
            order.status = ExecutionStatusV2::PartiallyFilled;
            order.unresolved_reason = None;
        }

        self.fills.insert(key, incoming);
        if contradictory_after_cancel || status_before == ExecutionStatusV2::ReconciliationRequired
        {
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        }
        else
        {
            Ok(ApplyOutcomeV2::Applied)
        }
    }

    fn apply_amended(
        &mut self,
        client_order_id: &str,
        request: ExactOrderRequest,
        received_at_ms: i64,
    ) -> Result<ApplyOutcomeV2, LifecycleV2Error> {
        let order = self.order_mut(client_order_id)?;
        if !matches!(
            order.status,
            ExecutionStatusV2::PendingAmend | ExecutionStatusV2::AmendUnknown
        ) || request.instrument_id != order.request.instrument_id
            || request.rules_version != order.request.rules_version
            || request.side != order.request.side
            || SignedAmount::from(request.quantity) < order.filled_quantity
        {
            return Err(LifecycleV2Error::InvalidAmend);
        }

        order.request = request;
        order.revision = order
            .revision
            .checked_add(1)
            .ok_or(LifecycleV2Error::ArithmeticOverflow)?;
        order.status = economic_open_status(order)?;
        order.unresolved_reason = None;
        order.last_event_ts_ms = received_at_ms;
        Ok(ApplyOutcomeV2::Applied)
    }

    fn order_mut(
        &mut self,
        client_order_id: &str,
    ) -> Result<&mut ExecutionOrderV2, LifecycleV2Error> {
        self.orders
            .get_mut(client_order_id)
            .ok_or_else(|| LifecycleV2Error::UnknownClientOrderId(client_order_id.to_string()))
    }
}

fn invalid_transition(order: &ExecutionOrderV2) -> LifecycleV2Error {
    LifecycleV2Error::InvalidTransition {
        client_order_id: order.client_order_id.clone(),
        status: order.status,
    }
}

fn economic_open_status(order: &ExecutionOrderV2) -> Result<ExecutionStatusV2, LifecycleV2Error> {
    if order.remaining_quantity()?.is_zero()
    {
        Ok(ExecutionStatusV2::Filled)
    }
    else if order.filled_quantity.is_zero()
    {
        Ok(ExecutionStatusV2::Open)
    }
    else
    {
        Ok(ExecutionStatusV2::PartiallyFilled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::financial::ExactOrderType;
    use crate::orders::{Side, TimeInForce};

    fn p(value: &str) -> Price {
        Price::parse(value).unwrap()
    }

    fn q(value: &str) -> Quantity {
        Quantity::parse(value).unwrap()
    }

    fn amount(value: &str) -> SignedAmount {
        SignedAmount::parse(value).unwrap()
    }

    fn request(quantity: &str) -> ExactOrderRequest {
        ExactOrderRequest {
            instrument_id: "BTC-USDT".into(),
            side: Side::Buy,
            order_type: ExactOrderType::Limit { price: p("100") },
            quantity: q(quantity),
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: false,
            rules_version: "rules-1".into(),
        }
    }

    fn book(quantity: &str) -> LifecycleBookV2 {
        let mut book = LifecycleBookV2::new("x", "acct-1").unwrap();
        book.register_intent(
            "intent-1".into(),
            "idem-1".into(),
            "agent-1".into(),
            "client-1".into(),
            request(quantity),
            90,
        )
        .unwrap();
        book
    }

    fn event(sequence: u64, kind: ExecutionEventKindV2) -> ExecutionEventV2 {
        ExecutionEventV2 {
            local_sequence: sequence,
            native_sequence: Some(sequence * 10),
            received_at_ms: 100 + sequence as i64,
            client_order_id: "client-1".into(),
            kind,
        }
    }

    fn fill(
        sequence: u64,
        trade_id: &str,
        occurred_at_ms: i64,
        quantity: &str,
        fee: &str,
    ) -> ExecutionEventV2 {
        event(
            sequence,
            ExecutionEventKindV2::Fill {
                key: FillKey {
                    venue: "x".into(),
                    account_id: "acct-1".into(),
                    instrument_id: "BTC-USDT".into(),
                    trade_id: trade_id.into(),
                },
                occurred_at_ms,
                price: p("100"),
                quantity: q(quantity),
                fee_asset: "BNB".into(),
                fee_amount: amount(fee),
            },
        )
    }

    fn accepted(sequence: u64) -> ExecutionEventV2 {
        event(
            sequence,
            ExecutionEventKindV2::Accepted {
                exchange_order_id: "exchange-1".into(),
            },
        )
    }

    #[test]
    fn local_sequence_is_contiguous_while_native_sequence_is_venue_evidence() {
        let mut book = book("2");
        let mut first = accepted(1);
        first.native_sequence = Some(100);
        book.apply_event(first).unwrap();

        let mut second = fill(2, "trade-1", 105, "0.5", "0.001");
        second.native_sequence = Some(900);
        assert_eq!(book.apply_event(second), Ok(ApplyOutcomeV2::Applied));
        assert_eq!(
            book.apply_event(fill(4, "trade-2", 106, "0.5", "0.001")),
            Err(LifecycleV2Error::LocalSequenceGap {
                expected: 3,
                incoming: 4,
            })
        );
    }

    #[test]
    fn duplicate_fill_with_new_transport_sequence_has_one_economic_effect() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(fill(2, "trade-1", 105, "0.5", "0.001"))
            .unwrap();
        let duplicate = book
            .apply_event(fill(3, "trade-1", 105, "0.5", "0.001"))
            .unwrap();
        assert_eq!(duplicate, ApplyOutcomeV2::DuplicateFill);
        assert_eq!(
            book.orders["client-1"].filled_quantity.as_decimal_string(),
            "0.5"
        );
        assert_eq!(
            book.orders["client-1"].fees_by_asset["BNB"].as_decimal_string(),
            "0.001"
        );
        assert_eq!(book.fills.len(), 1);
    }

    #[test]
    fn same_trade_identity_with_changed_economics_is_rejected() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(fill(2, "trade-1", 105, "0.5", "0.001"))
            .unwrap();
        assert_eq!(
            book.apply_event(fill(3, "trade-1", 105, "0.6", "0.001")),
            Err(LifecycleV2Error::FillIdentityConflict(FillKey {
                venue: "x".into(),
                account_id: "acct-1".into(),
                instrument_id: "BTC-USDT".into(),
                trade_id: "trade-1".into(),
            }))
        );
    }

    #[test]
    fn fill_before_ack_is_accounted_and_late_ack_only_attaches_identity() {
        let mut book = book("2");
        book.apply_event(fill(1, "trade-1", 100, "0.5", "0.001"))
            .unwrap();
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::PartiallyFilled
        );
        book.apply_event(accepted(2)).unwrap();
        assert_eq!(
            book.orders["client-1"].exchange_order_id.as_deref(),
            Some("exchange-1")
        );
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::PartiallyFilled
        );
    }

    #[test]
    fn submit_unknown_never_implies_safe_resubmit() {
        let mut book = book("2");
        assert_eq!(
            book.apply_event(event(
                1,
                ExecutionEventKindV2::SubmitUnknown {
                    reason: "timeout".into(),
                },
            )),
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        );
        let order = &book.orders["client-1"];
        assert_eq!(order.status, ExecutionStatusV2::SubmitUnknown);
        assert!(!order.may_repeat_submit());
    }

    #[test]
    fn rejection_after_fill_is_contradictory_evidence_not_erasure() {
        let mut book = book("2");
        book.apply_event(fill(1, "trade-1", 100, "0.5", "0.001"))
            .unwrap();
        assert_eq!(
            book.apply_event(event(
                2,
                ExecutionEventKindV2::SubmitRejected {
                    reason: "venue says rejected".into(),
                },
            )),
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        );
        assert_eq!(
            book.orders["client-1"].filled_quantity.as_decimal_string(),
            "0.5"
        );
    }

    #[test]
    fn reconciliation_required_is_sticky_across_later_fills() {
        let mut book = book("2");
        book.apply_event(fill(1, "trade-1", 100, "0.5", "0.001"))
            .unwrap();
        book.apply_event(event(
            2,
            ExecutionEventKindV2::SubmitRejected {
                reason: "venue says rejected".into(),
            },
        ))
        .unwrap();
        let reason = book.orders["client-1"].unresolved_reason.clone();
        assert_eq!(
            book.apply_event(fill(3, "trade-2", 110, "0.5", "0.001")),
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        );
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::ReconciliationRequired
        );
        assert_eq!(book.orders["client-1"].unresolved_reason, reason);
        assert_eq!(
            book.orders["client-1"].filled_quantity.as_decimal_string(),
            "1"
        );
    }

    #[test]
    fn cancel_rejection_returns_to_economic_open_state() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(fill(2, "trade-1", 105, "0.5", "0.001"))
            .unwrap();
        book.apply_event(event(3, ExecutionEventKindV2::CancelRequested))
            .unwrap();
        book.apply_event(event(
            4,
            ExecutionEventKindV2::CancelRejected {
                reason: "too late".into(),
            },
        ))
        .unwrap();
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::PartiallyFilled
        );
    }

    #[test]
    fn late_delivery_of_pre_cancel_fill_is_accounted() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(event(2, ExecutionEventKindV2::CancelRequested))
            .unwrap();
        book.apply_event(event(
            3,
            ExecutionEventKindV2::Canceled {
                effective_at_ms: 150,
            },
        ))
        .unwrap();
        book.apply_event(fill(4, "trade-1", 140, "0.5", "0.001"))
            .unwrap();
        assert_eq!(
            book.orders["client-1"].filled_quantity.as_decimal_string(),
            "0.5"
        );
        assert_eq!(book.orders["client-1"].status, ExecutionStatusV2::Canceled);
    }

    #[test]
    fn fill_occurring_after_confirmed_cancel_requires_reconciliation() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(event(2, ExecutionEventKindV2::CancelRequested))
            .unwrap();
        book.apply_event(event(
            3,
            ExecutionEventKindV2::Canceled {
                effective_at_ms: 150,
            },
        ))
        .unwrap();
        assert_eq!(
            book.apply_event(fill(4, "trade-1", 160, "0.5", "0.001")),
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        );
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::ReconciliationRequired
        );
    }

    #[test]
    fn later_cancel_confirmation_rechecks_already_recorded_fills() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(event(2, ExecutionEventKindV2::CancelRequested))
            .unwrap();
        book.apply_event(event(
            3,
            ExecutionEventKindV2::CancelUnknown {
                reason: "timeout".into(),
            },
        ))
        .unwrap();
        book.apply_event(fill(4, "trade-1", 160, "0.5", "0.001"))
            .unwrap();
        assert_eq!(
            book.apply_event(event(
                5,
                ExecutionEventKindV2::Canceled {
                    effective_at_ms: 150,
                },
            )),
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        );
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::ReconciliationRequired
        );
    }

    #[test]
    fn exact_fill_accounting_has_no_epsilon_overfill_rule() {
        let mut exact = book("0.3");
        exact.apply_event(accepted(1)).unwrap();
        exact
            .apply_event(fill(2, "trade-1", 110, "0.1", "0"))
            .unwrap();
        exact
            .apply_event(fill(3, "trade-2", 120, "0.2", "0"))
            .unwrap();
        assert_eq!(exact.orders["client-1"].status, ExecutionStatusV2::Filled);

        let mut overfill = book("0.3");
        overfill.apply_event(accepted(1)).unwrap();
        overfill
            .apply_event(fill(2, "trade-1", 110, "0.1", "0"))
            .unwrap();
        assert_eq!(
            overfill.apply_event(fill(3, "trade-2", 120, "0.200000000000000001", "0",)),
            Err(LifecycleV2Error::Overfill)
        );
    }

    #[test]
    fn fee_overflow_does_not_partially_apply_fill_quantity() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        let max = "170141183460469231731687303715884105727";
        book.apply_event(fill(2, "trade-1", 110, "0.5", max))
            .unwrap();
        assert_eq!(
            book.apply_event(fill(3, "trade-2", 120, "0.5", "1")),
            Err(LifecycleV2Error::ArithmeticOverflow)
        );
        assert_eq!(book.last_local_sequence, 2);
        assert_eq!(book.fills.len(), 1);
        assert_eq!(
            book.orders["client-1"].filled_quantity.as_decimal_string(),
            "0.5"
        );
        assert_eq!(
            book.orders["client-1"].fees_by_asset["BNB"].as_decimal_string(),
            max
        );
    }

    #[test]
    fn fees_can_be_third_asset_and_signed_rebates() {
        let mut book = book("1");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(fill(2, "trade-1", 110, "0.5", "0.001"))
            .unwrap();
        book.apply_event(fill(3, "trade-2", 120, "0.5", "-0.0002"))
            .unwrap();
        assert_eq!(
            book.orders["client-1"].fees_by_asset["BNB"].as_decimal_string(),
            "0.0008"
        );
    }

    #[test]
    fn full_fill_preserves_pending_amend_until_confirmation() {
        let mut book = book("1");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(event(2, ExecutionEventKindV2::AmendRequested))
            .unwrap();
        book.apply_event(fill(3, "trade-1", 110, "1", "0")).unwrap();
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::PendingAmend
        );
        book.apply_event(event(
            4,
            ExecutionEventKindV2::Amended {
                request: request("2"),
            },
        ))
        .unwrap();
        assert_eq!(
            book.orders["client-1"].status,
            ExecutionStatusV2::PartiallyFilled
        );
        assert_eq!(
            book.orders["client-1"]
                .remaining_quantity()
                .unwrap()
                .as_decimal_string(),
            "1"
        );
    }

    #[test]
    fn amend_unknown_then_rejected_returns_to_open_state() {
        let mut book = book("2");
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(event(2, ExecutionEventKindV2::AmendRequested))
            .unwrap();
        assert_eq!(
            book.apply_event(event(
                3,
                ExecutionEventKindV2::AmendUnknown {
                    reason: "timeout".into(),
                },
            )),
            Ok(ApplyOutcomeV2::ReconciliationRequired)
        );
        book.apply_event(event(
            4,
            ExecutionEventKindV2::AmendRejected {
                reason: "not applied".into(),
            },
        ))
        .unwrap();
        assert_eq!(book.orders["client-1"].status, ExecutionStatusV2::Open);
    }
}
